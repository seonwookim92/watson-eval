import os
import sys
import glob
import hashlib
import pickle
import rdflib
import numpy as np
import requests as _requests
from rdflib.namespace import RDF, RDFS, OWL
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

SHACL = rdflib.Namespace("http://www.w3.org/ns/shacl#")

# Bump this whenever the cache format or schema parsing logic changes
_CACHE_VERSION = "3"


class _RemoteEmbeddingModel:
    """Minimal SentenceTransformer-compatible client for a remote OpenAI-compatible embedding API."""

    def __init__(self, api_url: str, model: str, api_key: str = "no-key"):
        self._api_url = api_url
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def encode(self, text, batch_size=None, show_progress_bar=False, **kwargs):
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        resp = _requests.post(
            self._api_url,
            headers=self._headers,
            json={"input": texts, "model": self._model, "truncate_prompt_tokens": 256},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        vecs = [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
        result = np.array(vecs)
        return result[0] if single else result


class _LocalEmbeddingModel:
    """SentenceTransformer-backed local embedding model."""

    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)

    def encode(self, text, batch_size=None, show_progress_bar=False, **kwargs):
        return self._model.encode(
            text,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            **kwargs,
        )


class OntologyEngine:
    """Core ontology management engine with semantic search capabilities."""
    def __init__(self, ontology_path, model_name="all-MiniLM-L6-v2"):
        self.graph = rdflib.Graph()
        self.ontology_path = ontology_path
        self.classes = {}
        self.properties = {}
        self.shapes = {}
        self.instance_base_uri = "http://example.org/entities/"
        self._ontology_files = []  # [(filepath, format), ...] — used for cache key

        embedding_mode = os.environ.get("EMBEDDING_MODE", "local").strip().lower()
        if embedding_mode == "remote":
            api_url = os.environ.get("EMBEDDING_API_URL", "http://192.168.100.2:8082/v1/embeddings")
            api_key = os.environ.get("EMBEDDING_API_KEY", "no-key")
            sys.stderr.write(
                f"Initializing Remote Embedding Model ({model_name} @ {api_url})...\n"
            )
            self.model = _RemoteEmbeddingModel(api_url, model_name, api_key)
        else:
            sys.stderr.write(f"Initializing Local Embedding Model ({model_name})...\n")
            self.model = _LocalEmbeddingModel(model_name)
        self.load_ontology()

    @staticmethod
    def _detect_format(filepath):
        """Detects rdflib parse format from file extension."""
        ext = os.path.splitext(filepath)[1].lower()
        return {
            ".ttl": "ttl",
            ".owl": "xml",
            ".rdf": "xml",
            ".n3":  "n3",
            ".nt":  "nt",
            ".trig": "trig",
            ".jsonld": "json-ld",
        }.get(ext, None)

    def load_ontology(self):
        """Loads all Turtle (.ttl) and OWL/RDF-XML (.owl, .rdf) files and parses the schema."""
        import contextlib
        with contextlib.redirect_stdout(sys.stderr):
            files = []

            if os.path.isdir(self.ontology_path):
                for pattern, fmt in [("**/*.ttl", "ttl"), ("**/*.owl", "xml"), ("**/*.rdf", "xml")]:
                    for f in glob.glob(os.path.join(self.ontology_path, pattern), recursive=True):
                        files.append((f, fmt))
            elif os.path.isfile(self.ontology_path):
                fmt = self._detect_format(self.ontology_path)
                if fmt:
                    files = [(self.ontology_path, fmt)]
                else:
                    sys.stderr.write(f"Warning: unrecognized file extension for {self.ontology_path}, trying ttl\n")
                    files = [(self.ontology_path, "ttl")]

            self._ontology_files = files  # Store for embedding cache key computation

            counts = {"ttl": 0, "xml": 0, "errors": 0}
            for f, fmt in files:
                try:
                    self.graph.parse(f, format=fmt)
                    counts[fmt] = counts.get(fmt, 0) + 1
                except Exception as e:
                    counts["errors"] += 1
                    sys.stderr.write(f"Error loading {f}: {e}\n")

            total = sum(v for k, v in counts.items() if k != "errors")
            detail = ", ".join(
                f"{v} {k.upper() if k != 'xml' else 'OWL/RDF-XML'}"
                for k, v in counts.items()
                if k != "errors" and v > 0
            )
            sys.stderr.write(
                f"Ontology loaded: {total} files ({detail})"
                + (f", {counts['errors']} errors" if counts['errors'] else "") + "\n"
            )
            self._build_schema_cache()

    def _build_schema_cache(self):
        """Organizes ontology items into memory. Embeddings are computed/loaded separately."""
        # 1. Extract Class Information (named URIs only — skip OWL anonymous blank nodes)
        named_classes = {
            s for s in
            set(self.graph.subjects(RDF.type, OWL.Class)) | set(self.graph.subjects(RDF.type, RDFS.Class))
            if isinstance(s, rdflib.URIRef)
        }
        for s in named_classes:
            uri = str(s)
            name = str(self.graph.value(s, RDFS.label) or uri.split('/')[-1].split('#')[-1])
            comment = str(self.graph.value(s, RDFS.comment) or "No description")
            self.classes[uri] = {
                'name': name,
                'comment': comment,
                'superclasses': [],
                'subclasses': [],
                # Embeddings populated by _build_embeddings()
            }

        # 2. Build Hierarchy
        for s, p, o in self.graph.triples((None, RDFS.subClassOf, None)):
            s_uri, o_uri = str(s), str(o)
            if s_uri in self.classes and o_uri in self.classes:
                self.classes[s_uri]['superclasses'].append(o_uri)
                self.classes[o_uri]['subclasses'].append(s_uri)

        # 3. Extract Property Information (named URIs only)
        OWL_DATA_PROPERTY = rdflib.URIRef("http://www.w3.org/2002/07/owl#DataProperty")

        for p_uri in set(self.graph.subjects(RDF.type, None)):
            if not isinstance(p_uri, rdflib.URIRef):
                continue
            p_type = self.graph.value(p_uri, RDF.type)

            # Support DatatypeProperty, DataProperty, ObjectProperty, and generic RDF.Property
            if p_type in [OWL.ObjectProperty, OWL.DatatypeProperty, OWL_DATA_PROPERTY, RDF.Property]:
                uri_str = str(p_uri)
                name = str(self.graph.value(p_uri, RDFS.label) or uri_str.split('/')[-1].split('#')[-1])
                comment = str(self.graph.value(p_uri, RDFS.comment) or "")

                # Normalize type name (DataProperty -> DatatypeProperty)
                type_name = str(p_type).split('/')[-1].split('#')[-1]
                if type_name == "DataProperty":
                    type_name = "DatatypeProperty"

                self.properties[uri_str] = {
                    'name': name,
                    'comment': comment,
                    'type': type_name,
                    'domain': [str(d) for d in self.graph.objects(p_uri, RDFS.domain)],
                    'range': [str(r) for r in self.graph.objects(p_uri, RDFS.range)],
                    # Embeddings populated by _build_embeddings()
                }

        # 4. Extract SHACL Shape Constraints and augment property domains
        for shape in self.graph.subjects(RDF.type, SHACL.NodeShape):
            target_class = self.graph.value(shape, SHACL.targetClass)
            if not target_class:
                continue
            target_uri = str(target_class)
            if target_uri not in self.shapes:
                self.shapes[target_uri] = []
            for prop_shape in self.graph.objects(shape, SHACL.property):
                path = self.graph.value(prop_shape, SHACL.path)
                if not path:
                    continue
                path_uri = str(path)
                # Add inferred domain to property info
                if path_uri in self.properties and target_uri not in self.properties[path_uri]['domain']:
                    self.properties[path_uri]['domain'].append(target_uri)

                # Resolve sh:node -> sh:targetClass for facet discovery
                node_target = None
                node_shape_ref = self.graph.value(prop_shape, SHACL.node)
                if node_shape_ref:
                    node_class = self.graph.value(node_shape_ref, SHACL.targetClass)
                    if node_class:
                        node_target = str(node_class)

                self.shapes[target_uri].append({
                    'path': path_uri,
                    'minCount': self.graph.value(prop_shape, SHACL.minCount),
                    'maxCount': self.graph.value(prop_shape, SHACL.maxCount),
                    'datatype': self.graph.value(prop_shape, SHACL.datatype),
                    'node_target': node_target,  # resolved from sh:node -> sh:targetClass
                })

        # 5. Compute or restore embeddings (with on-disk caching)
        self._build_embeddings()

    # ── Embedding Cache ──────────────────────────────────────────────────────

    def _get_cache_path(self) -> str:
        """Returns the path for the embedding cache file."""
        if os.path.isdir(self.ontology_path):
            return os.path.join(self.ontology_path, ".embedding_cache.pkl")
        return self.ontology_path + ".embedding_cache.pkl"

    def _compute_cache_key(self) -> str:
        """Computes a cache key from the loaded ontology files' paths, sizes, and mtimes."""
        h = hashlib.md5()
        for filepath, _ in sorted(self._ontology_files):
            try:
                stat = os.stat(filepath)
                h.update(f"{filepath}:{stat.st_size}:{stat.st_mtime_ns}".encode())
            except OSError:
                pass
        return h.hexdigest()

    def _load_embedding_cache(self, cache_path: str, cache_key: str) -> bool:
        """Tries to restore embeddings from cache. Returns True on cache hit."""
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            if cached.get('cache_key') != cache_key or cached.get('version') != _CACHE_VERSION:
                return False
            for uri, embs in cached.get('class_embeddings', {}).items():
                if uri in self.classes:
                    self.classes[uri]['name_embedding'] = embs['name']
                    self.classes[uri]['comment_embedding'] = embs['comment']
            for uri, embs in cached.get('property_embeddings', {}).items():
                if uri in self.properties:
                    self.properties[uri]['name_embedding'] = embs['name']
                    self.properties[uri]['comment_embedding'] = embs['comment']
            # Reject partial cache (e.g., schema changed but mtime didn't)
            if any('name_embedding' not in v for v in self.classes.values()):
                return False
            if any('name_embedding' not in v for v in self.properties.values()):
                return False
            return True
        except Exception:
            return False

    def _save_embedding_cache(self, cache_path: str, cache_key: str):
        """Persists current embeddings to disk."""
        try:
            cached = {
                'version': _CACHE_VERSION,
                'cache_key': cache_key,
                'class_embeddings': {
                    uri: {'name': info['name_embedding'], 'comment': info['comment_embedding']}
                    for uri, info in self.classes.items()
                },
                'property_embeddings': {
                    uri: {'name': info['name_embedding'], 'comment': info['comment_embedding']}
                    for uri, info in self.properties.items()
                },
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cached, f)
            sys.stderr.write(f"Embedding cache saved to {cache_path}\n")
        except Exception as e:
            sys.stderr.write(f"Warning: Could not save embedding cache: {e}\n")

    def _build_embeddings(self):
        """Loads embeddings from cache, or batch-encodes and saves them."""
        cache_path = self._get_cache_path()
        cache_key = self._compute_cache_key()

        if self._load_embedding_cache(cache_path, cache_key):
            sys.stderr.write("Embedding cache hit — skipping re-encoding.\n")
            return

        sys.stderr.write("Embedding cache miss — batch-encoding all terms...\n")

        class_uris = list(self.classes.keys())
        prop_uris = list(self.properties.keys())

        # Collect all texts and batch-encode in a single pass (much faster than one-by-one)
        texts = (
            [self.classes[u]['name']    for u in class_uris]
            + [self.classes[u]['comment'] for u in class_uris]
            + [self.properties[u]['name']    for u in prop_uris]
            + [self.properties[u]['comment'] for u in prop_uris]
        )

        embeddings = self.model.encode(texts, batch_size=64, show_progress_bar=False) if texts else []

        n_c = len(class_uris)
        n_p = len(prop_uris)

        for i, uri in enumerate(class_uris):
            self.classes[uri]['name_embedding']    = embeddings[i]
            self.classes[uri]['comment_embedding'] = embeddings[n_c + i]

        for i, uri in enumerate(prop_uris):
            self.properties[uri]['name_embedding']    = embeddings[2 * n_c + i]
            self.properties[uri]['comment_embedding'] = embeddings[2 * n_c + n_p + i]

        self._save_embedding_cache(cache_path, cache_key)

    # ── Schema Queries ───────────────────────────────────────────────────────

    def get_properties_for_class(self, class_uri: str, property_type: str = None, include_facets: bool = False) -> list[str]:
        """Returns all valid properties for a given class, including inherited ones.

        Args:
            class_uri: The URI of the class.
            property_type: Optional filter ('ObjectProperty', 'DatatypeProperty').
            include_facets: If True, also includes properties of related Facet classes.
        """
        relevant_classes = {class_uri} | set(self.get_transitive_superclasses(class_uri))
        if include_facets:
            for facet_uri in self.get_related_facets(class_uri):
                relevant_classes.add(facet_uri)
                relevant_classes.update(self.get_transitive_superclasses(facet_uri))

        valid_props = []
        for p_uri, info in self.properties.items():
            if property_type and info['type'] != property_type:
                continue
            if not info['domain'] or any(d in relevant_classes for d in info['domain']):
                valid_props.append(p_uri)
        return list(set(valid_props))

    def get_related_facets(self, class_uri: str) -> list[str]:
        """
        Finds related Facet classes for a given class using four strategies:

        1. Direct hasFacet-like property range (name heuristic)
        2. rdfs:subPropertyOf — properties extending any hasFacet-like property
        3. SHACL sh:node — shape constraint resolving to a concrete target class
        4. SHACL path name heuristic — shape entries whose path contains 'hasFacet'
        """
        facets = set()
        relevant_classes = {class_uri} | set(self.get_transitive_superclasses(class_uri))

        # Identify all hasFacet-like root properties by name
        hasfacet_prop_uris = {
            p_uri for p_uri in self.properties
            if "hasFacet" in p_uri or "hasfacet" in p_uri.lower()
        }

        # 1. Direct range of hasFacet-like properties whose domain includes this class
        for p_uri in hasfacet_prop_uris:
            info = self.properties[p_uri]
            if any(d in relevant_classes for d in info["domain"]):
                facets.update(info["range"])

        # 2. rdfs:subPropertyOf — sub-properties of any hasFacet-like property
        for p_uri, info in self.properties.items():
            p_ref = rdflib.URIRef(p_uri)
            for super_prop in self.graph.objects(p_ref, RDFS.subPropertyOf):
                if str(super_prop) in hasfacet_prop_uris:
                    if any(d in relevant_classes for d in info["domain"]):
                        facets.update(info["range"])

        # 3 & 4. SHACL shape entries for this class and its superclasses
        for target in relevant_classes:
            if target not in self.shapes:
                continue
            for shape_entry in self.shapes[target]:
                path = shape_entry['path']
                is_facet_path = "hasFacet" in path or "hasfacet" in path.lower()

                # Strategy 3: sh:node resolved to a concrete target class
                if shape_entry.get('node_target') and is_facet_path:
                    facets.add(shape_entry['node_target'])
                # Strategy 4: fallback to property range via path name
                elif is_facet_path:
                    p_info = self.properties.get(path)
                    if p_info:
                        facets.update(p_info["range"])

        return list(facets)

    def infer_datatype(self, value: str) -> str | None:
        """Heuristic to infer XSD datatype from a string value."""
        if not value:
            return None
        import re
        if re.match(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?", value):
            return "http://www.w3.org/2001/XMLSchema#dateTime"
        if re.match(r"^-?\d+$", value):
            return "http://www.w3.org/2001/XMLSchema#integer"
        if value.lower() in ["true", "false"]:
            return "http://www.w3.org/2001/XMLSchema#boolean"
        return "http://www.w3.org/2001/XMLSchema#string"

    def get_candidate_relations(self, subject_types: list, object_types: list) -> list:
        """
        Finds all possible relations between two sets of types.
        Returns list of dicts: {uri, path_type: 'direct'|'facet'|'inverse', bridge: facet_uri_or_none}
        """
        candidates = []

        all_s_types = set()
        for st in subject_types:
            all_s_types.add(st)
            all_s_types.update(self.get_transitive_superclasses(st))

        all_o_types = set()
        for ot in object_types:
            all_o_types.add(ot)
            all_o_types.update(self.get_transitive_superclasses(ot))

        # 1. Direct & Facet-based Forward Relations
        for st in subject_types:
            direct_props = self.get_properties_for_class(st, "ObjectProperty", include_facets=False)
            for p_uri in direct_props:
                p_range = self.properties[p_uri]['range']
                if not p_range or any(r in all_o_types for r in p_range):
                    candidates.append({'uri': p_uri, 'path_type': 'direct', 'bridge': None})

            facets = self.get_related_facets(st)
            for f_uri in facets:
                f_props = self.get_properties_for_class(f_uri, "ObjectProperty", include_facets=False)
                for p_uri in f_props:
                    p_range = self.properties[p_uri]['range']
                    if not p_range or any(r in all_o_types for r in p_range):
                        candidates.append({'uri': p_uri, 'path_type': 'facet', 'bridge': f_uri})

        # 2. Inverse Relations (Object -> Subject)
        for ot in object_types:
            inv_props = self.get_properties_for_class(ot, "ObjectProperty", include_facets=True)
            for p_uri in inv_props:
                p_range = self.properties[p_uri]['range']
                if any(r in all_s_types for r in p_range):
                    if not any(c['uri'] == p_uri for c in candidates):
                        candidates.append({'uri': p_uri, 'path_type': 'inverse', 'bridge': None})

        return candidates

    def semantic_similarity(self, query, embeddings):
        """Calculates cosine similarity."""
        query_vec = self.model.encode(query, show_progress_bar=False)
        scores = []
        for emb in embeddings:
            norm_q = np.linalg.norm(query_vec)
            norm_e = np.linalg.norm(emb)
            if norm_q < 1e-9 or norm_e < 1e-9:
                scores.append(0.0)
            else:
                scores.append(float(np.dot(query_vec, emb) / (norm_q * norm_e)))
        return scores

    def weighted_similarity(self, query, name_embeddings, comment_embeddings, weights=(0.5, 0.5)):
        """Calculates weighted similarity between query and (name, comment) pairs."""
        query_vec = self.model.encode(query, show_progress_bar=False)
        q_norm = np.linalg.norm(query_vec)
        if q_norm < 1e-9:
            return [0.0] * len(name_embeddings)

        scores = []
        for n_emb, c_emb in zip(name_embeddings, comment_embeddings):
            n_norm = np.linalg.norm(n_emb)
            c_norm = np.linalg.norm(c_emb)
            n_sim = np.dot(query_vec, n_emb) / (q_norm * n_norm) if n_norm > 1e-9 else 0.0
            c_sim = np.dot(query_vec, c_emb) / (q_norm * c_norm) if c_norm > 1e-9 else 0.0
            scores.append(float(n_sim * weights[0] + c_sim * weights[1]))
        return scores

    def get_transitive_superclasses(self, class_uri):
        """Returns the full inheritance tree upwards."""
        superclasses = set()
        stack = [class_uri]
        while stack:
            curr = stack.pop()
            parents = self.classes.get(curr, {}).get('superclasses', [])
            for p in parents:
                if p not in superclasses:
                    superclasses.add(p)
                    stack.append(p)
        return list(superclasses)
