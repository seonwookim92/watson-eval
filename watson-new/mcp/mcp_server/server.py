import json
import urllib.error
import uuid
import rdflib
import urllib.parse
import urllib.request
from typing import Any
from rdflib.namespace import RDF
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, TypeAdapter
from .engine import OntologyEngine
from .config import (
    MAPPING_GUIDELINES,
    DEFAULT_ONTOLOGY_DIR,
    EMBEDDING_MODEL,
    PROPERTY_RECOMMENDER_BASE_URL,
    PROPERTY_RECOMMENDER_API_KEY,
    PROPERTY_RECOMMENDER_MODEL,
    PROPERTY_RECOMMENDER_TIMEOUT_SECONDS,
)

import logging

# Score adjustment constants for search and ranking
KEYWORD_BONUS = 0.3            # Bonus when query is found verbatim in name/comment
SEMANTIC_MATCH_THRESHOLD = 0.6 # Minimum semantic score to include a property result
DATATYPE_MATCH_BOOST = 0.2     # Boost when property range matches inferred value type
DIRECT_RELATION_BOOST = 0.1    # Boost for directly defined object properties
INVERSE_RELATION_PENALTY = 0.05  # Penalty for inverse (reversed-direction) relations
PROPERTY_RECOMMENDER_RETRY_LIMIT = 3
PROPERTY_RECOMMENDER_CANDIDATES_PER_BUCKET = 8

# Silence noisy logs from FastMCP and underlying libraries
logging.basicConfig(level=logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Initialize FastMCP server
mcp = FastMCP("Universal Ontology Core", instructions=MAPPING_GUIDELINES)

# Global manager instance
engine = OntologyEngine(DEFAULT_ONTOLOGY_DIR, model_name=EMBEDDING_MODEL)


class RecommendedProperty(BaseModel):
    isReverse: bool
    isDataProperty: bool
    propertyURI: str
    confidence: float = Field(ge=0.0, le=1.0)


class RecommendPropertyResponse(BaseModel):
    isSuccess: bool
    result: list[RecommendedProperty]


RECOMMENDED_PROPERTY_LIST_ADAPTER = TypeAdapter(list[RecommendedProperty])


def _short_uri(uri: str) -> str:
    return uri.split("/")[-1].split("#")[-1]


def _combine_query_and_context(query: str, context: str | None) -> str:
    query = query.strip()
    if context and context.strip():
        return f"{query} {context.strip()}"
    return query


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _extract_json_array(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found in the LLM response.")
    return text[start:end + 1]


def _call_property_recommender_llm(messages: list[dict[str, str]]) -> str:
    if not PROPERTY_RECOMMENDER_MODEL:
        raise RuntimeError("PROPERTY_RECOMMENDER_MODEL is not configured.")

    payload = {
        "model": PROPERTY_RECOMMENDER_MODEL,
        "messages": messages,
        "temperature": 0,
    }
    headers = {"Content-Type": "application/json"}
    if PROPERTY_RECOMMENDER_API_KEY:
        headers["Authorization"] = f"Bearer {PROPERTY_RECOMMENDER_API_KEY}"

    request = urllib.request.Request(
        f"{PROPERTY_RECOMMENDER_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=PROPERTY_RECOMMENDER_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected LLM response payload: {body}") from exc

    return _extract_text_content(content)


def _build_property_candidate_description(candidate: dict[str, Any], subject_type_uri: str, object_type_uri: str) -> str:
    prop_info = engine.properties[candidate["uri"]]
    bridge_name = _short_uri(candidate["bridge"]) if candidate.get("bridge") else None

    if candidate["is_data_property"]:
        owner_uri = object_type_uri if candidate["is_reverse"] else subject_type_uri
        direction = f"{_short_uri(owner_uri)} -> Literal"
        if candidate["is_reverse"]:
            direction += " (reverse relative to input S/O)"
        selection_hint = f"Use when {_short_uri(owner_uri)} should carry a literal value matching the predicate."
    else:
        src_uri = object_type_uri if candidate["is_reverse"] else subject_type_uri
        dst_uri = subject_type_uri if candidate["is_reverse"] else object_type_uri
        direction = f"{_short_uri(src_uri)} -> {_short_uri(dst_uri)}"
        selection_hint = f"Use when the ontology relation should point from {_short_uri(src_uri)} to {_short_uri(dst_uri)}."

    if candidate["path_type"] == "facet" and bridge_name:
        selection_hint += f" This property is discovered via facet/component {bridge_name}."
    elif candidate["path_type"] == "direct":
        selection_hint += " This is a direct relation."

    return json.dumps(
        {
            "propertyURI": candidate["uri"],
            "propertyName": prop_info["name"],
            "description": prop_info["comment"] or "",
            "isReverse": candidate["is_reverse"],
            "isDataProperty": candidate["is_data_property"],
            "direction": direction,
            "pathType": candidate["path_type"],
            "bridgeClassURI": candidate["bridge"],
            "domainURIs": prop_info["domain"],
            "rangeURIs": prop_info["range"],
            "selectionHint": selection_hint,
            "retrievalScore": round(candidate["retrieval_score"], 4),
        },
        ensure_ascii=True,
    )


def _rank_recommend_property_candidates(
    candidates: list[dict[str, Any]],
    combined_query: str,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    candidate_uris = [candidate["uri"] for candidate in candidates]
    name_embs = [engine.properties[uri]["name_embedding"] for uri in candidate_uris]
    comment_embs = [engine.properties[uri]["comment_embedding"] for uri in candidate_uris]
    sim_scores = engine.weighted_similarity(combined_query, name_embs, comment_embs)
    query_lower = combined_query.lower()

    scored_candidates = []
    for candidate, score in zip(candidates, sim_scores):
        info = engine.properties[candidate["uri"]]
        if query_lower and (
            query_lower in info["name"].lower() or query_lower in info["comment"].lower()
        ):
            score += KEYWORD_BONUS
        if not candidate["is_data_property"] and candidate["path_type"] == "direct":
            score += DIRECT_RELATION_BOOST
        scored = dict(candidate)
        scored["retrieval_score"] = float(score)
        scored_candidates.append(scored)

    bucketed = {
        (False, False): [],
        (True, False): [],
        (False, True): [],
        (True, True): [],
    }
    for candidate in scored_candidates:
        bucketed[(candidate["is_reverse"], candidate["is_data_property"])].append(candidate)

    selected = []
    for bucket in bucketed.values():
        bucket.sort(key=lambda item: item["retrieval_score"], reverse=True)
        selected.extend(bucket[:PROPERTY_RECOMMENDER_CANDIDATES_PER_BUCKET])

    selected.sort(key=lambda item: item["retrieval_score"], reverse=True)
    return selected


def _build_recommend_property_candidates(subject_type_uri: str, object_type_uri: str) -> list[dict[str, Any]]:
    raw_candidates = engine.get_property_candidates_for_type_pair(subject_type_uri, object_type_uri)
    candidates = []
    for candidate in raw_candidates:
        if candidate["uri"] not in engine.properties:
            continue
        candidates.append(candidate)
    return candidates


def _build_recommend_property_messages(
    subject_type_uri: str,
    object_type_uri: str,
    predicate: str,
    context: str | None,
    candidates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    candidate_lines = [
        _build_property_candidate_description(candidate, subject_type_uri, object_type_uri)
        for candidate in candidates
    ]

    system_prompt = (
        "You select ontology properties from a constrained candidate list.\n"
        "Return only a JSON array.\n"
        "Each array item must have exactly these keys: "
        "isReverse, isDataProperty, propertyURI, confidence.\n"
        "Rules:\n"
        "- Choose at most 3 candidates.\n"
        "- propertyURI must be one of the provided candidates.\n"
        "- Keep candidates ordered from most likely to least likely.\n"
        "- confidence must be a number between 0.0 and 1.0.\n"
        "- If none fit, return [].\n"
        "- isReverse=true means the correct triple direction is object -> subject relative to the input pair.\n"
        "- isDataProperty=true means the property connects an entity to a literal, not to the other entity."
    )
    user_prompt = (
        f"subject_type_uri: {subject_type_uri}\n"
        f"object_type_uri: {object_type_uri}\n"
        f"predicate: {predicate}\n"
        f"context: {context or ''}\n"
        "Candidates:\n"
        + "\n".join(candidate_lines)
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _validate_recommend_property_result(
    parsed: Any,
    candidates: list[dict[str, Any]],
) -> list[RecommendedProperty]:
    validated = RECOMMENDED_PROPERTY_LIST_ADAPTER.validate_python(parsed)
    if len(validated) > 3:
        raise ValueError("The LLM returned more than 3 recommendations.")

    candidate_lookup = {
        (candidate["uri"], candidate["is_reverse"], candidate["is_data_property"]): candidate
        for candidate in candidates
    }
    seen = set()
    for item in validated:
        key = (item.propertyURI, item.isReverse, item.isDataProperty)
        if key not in candidate_lookup:
            raise ValueError(f"Candidate not allowed: {item.propertyURI} / reverse={item.isReverse} / data={item.isDataProperty}")
        if key in seen:
            raise ValueError(f"Duplicate candidate returned: {item.propertyURI}")
        seen.add(key)
    return validated

# --- [Schema Inspection Tools] ---

@mcp.tool()
def list_root_classes() -> str:
    """Lists the top-level classes in the ontology (those with no parents)."""
    roots = []
    for uri, info in engine.classes.items():
        if not info['superclasses']:
            roots.append(f"- {info['name']} ({uri}): {info['comment']}")
    return "\n".join(sorted(roots)) or "No root classes found."

@mcp.tool()
def list_subclasses(class_uri: str) -> str:
    """Lists the immediate subclasses of a given class."""
    if class_uri not in engine.classes: return "Error: Class not found."
    subs = engine.classes[class_uri]['subclasses']
    result = []
    for uri in subs:
        cls = engine.classes[uri]
        result.append(f"- {cls['name']} ({uri}): {cls['comment']}")
    return "\n".join(sorted(result)) or "No subclasses found."

@mcp.tool()
def drill_into_classes(query: str, class_uri: str = None) -> str:
    """LLM-driven hierarchical class finder. Use this when search_classes scores are all below 0.5
    or the top result doesn't clearly match your concept.

    Unlike search_classes (embedding-based), this tool shows raw class names and descriptions
    at each level so YOU decide which branch to follow using your own judgment.
    No embeddings are computed — you read the descriptions and pick.

    Iterative workflow:
      1. Call with only `query` (no class_uri) → read root classes and descriptions.
      2. Pick the class whose name/description best matches what you seek.
      3. Call again with that class's URI as class_uri → go one level deeper.
      4. Repeat until you reach a leaf or find the right class.
      5. Confirm with get_class_details('<uri>') before using the class.

    Args:
        query: What you are looking for (reminder label only — not used for ranking).
        class_uri: Class to expand. Omit to start from root classes.
    """
    if class_uri:
        if class_uri not in engine.classes:
            return f"Error: Class '{class_uri}' not found."
        info = engine.classes[class_uri]
        candidate_uris = sorted(info['subclasses'], key=lambda u: engine.classes.get(u, {}).get('name', u))
        lines = [
            f"Current class: {info['name']}",
            f"URI: {class_uri}",
            f"Description: {info['comment']}",
            f"Looking for: '{query}'",
            "",
        ]
        if not candidate_uris:
            lines.append("This is a leaf class (no subclasses).")
            lines.append("→ If this matches what you need, use get_class_details to confirm its properties.")
            lines.append("→ If not, go back up and try a different branch.")
            return "\n".join(lines)
        lines.append(f"Subclasses ({len(candidate_uris)} total) — read descriptions and choose the best branch:")
    else:
        candidate_uris = sorted(
            [u for u, i in engine.classes.items() if not i['superclasses']],
            key=lambda u: engine.classes[u]['name']
        )
        lines = [
            f"Root classes — looking for: '{query}'",
            "Read the descriptions below and pick the branch most likely to contain your concept.",
            "",
            f"Root classes ({len(candidate_uris)} total):",
        ]

    lines.append("-" * 50)
    for uri in candidate_uris:
        cls = engine.classes[uri]
        n_children = len(cls['subclasses'])
        child_hint = f"[{n_children} subclasses]" if n_children else "[leaf]"
        lines.append(f"• {cls['name']}  {child_hint}")
        lines.append(f"  URI: {uri}")
        lines.append(f"  {cls['comment']}")
        lines.append("")

    lines.append("→ Next: drill_into_classes(query, '<chosen_uri>') to go one level deeper.")
    lines.append("→ Or: get_class_details('<uri>') to confirm a leaf or near-leaf match.")
    return "\n".join(lines)


@mcp.tool()
def show_class_tree(class_uri: str = None, depth: int = 2) -> str:
    """Displays the class hierarchy as an indented tree (read-only, no navigation).

    Use this to get a structural overview of the ontology or a specific subtree.
    For interactive top-down search, use drill_into_classes instead.

    Args:
        class_uri: Root of the subtree to display. If omitted, shows all top-level classes.
        depth: How many levels to expand (1–4 recommended). Branches beyond this depth
               show a '[N subclasses]' count instead of expanding further.
    """
    def _render(uri: str, current_depth: int, is_last: bool, prefix: str, visited: set) -> list[str]:
        if uri not in engine.classes or uri in visited:
            return []
        visited.add(uri)
        info = engine.classes[uri]
        connector = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")

        n_children = len(info['subclasses'])
        count_hint = f"  [{n_children} subclasses]" if n_children and current_depth >= depth else ""
        lines = [f"{prefix}{connector}{info['name']}{count_hint}  ({uri})"]

        if current_depth < depth and n_children:
            children = sorted(info['subclasses'], key=lambda u: engine.classes.get(u, {}).get('name', u))
            for i, child_uri in enumerate(children):
                lines.extend(_render(child_uri, current_depth + 1, i == len(children) - 1, child_prefix, visited))
        return lines

    # Determine starting nodes
    if class_uri:
        if class_uri not in engine.classes:
            return f"Error: Class '{class_uri}' not found."
        roots = [class_uri]
        header = f"Class tree from: {engine.classes[class_uri]['name']} (depth={depth})\n"
    else:
        roots = sorted(
            [u for u, info in engine.classes.items() if not info['superclasses']],
            key=lambda u: engine.classes[u]['name']
        )
        header = f"Top-level class tree (depth={depth})\n"

    lines = [header]
    visited: set = set()
    for i, root_uri in enumerate(roots):
        info = engine.classes.get(root_uri, {})
        n_children = len(info.get('subclasses', []))
        count_hint = f"  [{n_children} subclasses]" if n_children and depth == 0 else ""
        lines.append(f"{info.get('name', root_uri)}{count_hint}  ({root_uri})")
        if depth > 0:
            children = sorted(info.get('subclasses', []), key=lambda u: engine.classes.get(u, {}).get('name', u))
            for j, child_uri in enumerate(children):
                lines.extend(_render(child_uri, 1, j == len(children) - 1, "", visited))
        lines.append("")

    return "\n".join(lines)

@mcp.tool()
def get_class_hierarchy(class_uri: str) -> str:
    """Returns the inheritance path from the root to the specified class."""
    if class_uri not in engine.classes: return f"Class {class_uri} not found."
    supers = engine.get_transitive_superclasses(class_uri)
    path = [class_uri]
    curr = class_uri
    visited = {curr}
    while curr in engine.classes and engine.classes[curr]['superclasses']:
        curr = engine.classes[curr]['superclasses'][0]
        if curr in visited: break
        path.append(curr)
        visited.add(curr)
    return " -> ".join([engine.classes[u]['name'] for u in reversed(path) if u in engine.classes])

@mcp.tool()
def search_classes(query: str, limit: int = 5) -> str:
    """Search for classes using Hybrid Search (Keyword + Semantic Embeddings).

    IMPORTANT: Query must contain ONE concept only (e.g. "Vulnerability", "Organization", "Malware").
    Multi-concept queries like "Vulnerability Software Malware" dilute the embedding and return
    poor results. Call this tool once per concept, in parallel if possible.
    """
    class_uris = list(engine.classes.keys())
    name_embeddings = [engine.classes[u]['name_embedding'] for u in class_uris]
    comment_embeddings = [engine.classes[u]['comment_embedding'] for u in class_uris]
    sim_scores = engine.weighted_similarity(query, name_embeddings, comment_embeddings)
    
    final_scores = []
    query_lower = query.lower()
    for i, uri in enumerate(class_uris):
        cls = engine.classes[uri]
        base_score = sim_scores[i]
        keyword_bonus = KEYWORD_BONUS if query_lower in cls['name'].lower() or query_lower in cls['comment'].lower() else 0
        final_scores.append((base_score + keyword_bonus, uri))
    
    sorted_res = sorted(final_scores, key=lambda x: x[0], reverse=True)[:limit]
    if not sorted_res: return f"No results found for '{query}'."
    
    res = [f"Found {len(sorted_res)} semantic candidates for '{query}':"]
    for i, (score, uri) in enumerate(sorted_res):
        cls = engine.classes[uri]
        res.append(f"{i+1}. {cls['name']} (Sim: {score:.4f})")
        res.append(f"   URI: {uri}\n   Description: {cls['comment']}")
        if "facet" in cls['name'].lower():
            res.append(f"   💡 TIP: Use 'attach_component' for this facet.")
        res.append("")
    return "\n".join(res)

@mcp.tool()
def search_properties(query: str) -> str:
    """Searches for properties by keyword or semantic embedding.

    IMPORTANT: Query must contain ONE concept only (e.g. "version", "identifier", "name").
    Multi-concept queries dilute the embedding and return poor results.
    Call this tool once per property concept you are looking for.
    """
    prop_uris = list(engine.properties.keys())
    name_embeddings = [engine.properties[u]['name_embedding'] for u in prop_uris]
    comment_embeddings = [engine.properties[u]['comment_embedding'] for u in prop_uris]
    sim_scores = engine.weighted_similarity(query, name_embeddings, comment_embeddings)
    
    query_lower = query.lower()
    results = []
    for i, uri in enumerate(prop_uris):
        info = engine.properties[uri]
        score = sim_scores[i]
        match = query_lower in info['name'].lower() or query_lower in info['comment'].lower()
        if match or score > SEMANTIC_MATCH_THRESHOLD:
            results.append((score + (KEYWORD_BONUS if match else 0), uri, info))
            
    sorted_res = sorted(results, key=lambda x: x[0], reverse=True)[:5]
    if not sorted_res: return "No properties found."
    
    final = [f"Found {len(sorted_res)} property candidates for '{query}':"]
    for score, uri, info in sorted_res:
        # Simplify display
        domains = [d.split('/')[-1].split('#')[-1] for d in info['domain']][:5]
        ranges = [r.split('/')[-1].split('#')[-1] for r in info['range']][:3]
        
        final.append(f"- {info['name']} ({uri}) [Sim: {score:.4f}]")
        final.append(f"  Type: {info['type']}")
        final.append(f"  Valid for: {', '.join(domains) or 'Generic/Global'}")
        final.append(f"  Expected Value: {', '.join(ranges) or 'Any'}")
        
        # Add tip for Facets - only if similarity is high or domain is explicit
        if score > SEMANTIC_MATCH_THRESHOLD:
            for d in info['domain']:
                if "Facet" in d or "Component" in d:
                    facet_name = d.split('/')[-1].split('#')[-1]
                    final.append(f"  💡 TIP: This property belongs to [{facet_name}]. Use `attach_component` with this facet.")
                    break
        final.append("")
    return "\n".join(final)

@mcp.tool()
def recommend_attribute(entity_uri: str, query: str, value: str = None, context: str = None) -> str:
    """
    Recommends the best DatatypeProperty to set on an entity. Call this for EVERY literal
    value you want to attach after create_entity — this is the most efficient and accurate
    way to find the correct property. It is context-aware, facet-traversing, and
    domain/range-validated. Do NOT use search_properties as a substitute.

    - entity_uri: The URI of the already-created entity.
    - query: What the value represents (e.g. 'version number', 'identifier', 'timestamp').
    - value: The actual data value (e.g. '16.7', '2023-09-21'). Used for type inference.
    - context: The original source sentence or field name for better semantic matching.
    """
    subj_uri = rdflib.URIRef(entity_uri)
    types = [str(t) for t in engine.graph.objects(subj_uri, RDF.type)]
    
    if not types:
        return f"Error: No type defined for entity '{entity_uri}'. Please use create_entity first."
    
    # 1. Collect all valid DatatypeProperties for the entity's types (including facets)
    candidate_uris = []
    main_types = set()
    for t in types:
        main_types.add(t)
        main_types.update(engine.get_transitive_superclasses(t))
        candidate_uris.extend(engine.get_properties_for_class(t, "DatatypeProperty", include_facets=True))
    candidate_uris = list(set(candidate_uris))
    
    if not candidate_uris:
        return f"No valid DatatypeProperties found for the classes: {types}"

    # 2. Refine query with context if provided
    combined_query = query
    if context:
        combined_query = f"{query} {context}"
        
    # 3. Rank candidates by semantic similarity
    name_embs = [engine.properties[u]['name_embedding'] for u in candidate_uris]
    comment_embs = [engine.properties[u]['comment_embedding'] for u in candidate_uris]
    sim_scores = engine.weighted_similarity(combined_query, name_embs, comment_embs)
    
    # 4. Filter by Datatype (Heuristic)
    inferred_type = engine.infer_datatype(value) if value else None
    
    final_results = []
    for score, uri in zip(sim_scores, candidate_uris):
        info = engine.properties[uri]
        # Boost score if range matches inferred type
        if inferred_type and any(inferred_type in r for r in info['range']):
            score += DATATYPE_MATCH_BOOST
        final_results.append((score, uri))
    
    results = sorted(final_results, key=lambda x: x[0], reverse=True)[:5]
    
    res = [f"Recommended attributes for {entity_uri}"]
    if value: res.append(f"   Target Value: '{value}' (Inferred Type: {inferred_type.split('#')[-1] if inferred_type else 'Unknown'})")
    res.append(f"   Intent/Context: '{combined_query}'")
    res.append("-" * 40)

    for i, (score, uri) in enumerate(results):
        info = engine.properties[uri]
        res.append(f"{i+1}. {info['name']} ({uri}) [Score: {score:.4f}]")
        res.append(f"   Description: {info['comment'] or 'No description'}")
        
        # Facet guidance
        is_facet_prop = not any(d in main_types for d in info['domain'])
        if is_facet_prop and info['domain']:
            facet_name = info['domain'][0].split('/')[-1].split('#')[-1]
            res.append(f"   💡 TIP: This property belongs to [{facet_name}].")
            res.append(f"           Use `attach_component` or `create_entity` for this facet and link it to your main entity.")
        res.append("")
    
    return "\n".join(res)

@mcp.tool()
def recommend_relation(subject_uri: str, object_uri: str, query: str, context: str = None) -> str:
    """
    Recommends the best ObjectProperty to link two entities. Call this for EVERY relationship
    between entities after create_entity — this is the most efficient and accurate way to find
    the correct connecting property. It discovers direct, facet-bridged, and inverse relations
    automatically. Do NOT use search_properties as a substitute.

    - subject_uri: URI of the source entity (already created).
    - object_uri: URI of the target entity (already created).
    - query: The relationship intent (e.g. 'installed by', 'targets', 'patches', 'discovered').
    - context: The original source sentence for better semantic matching.
    """
    s_uri = rdflib.URIRef(subject_uri)
    o_uri = rdflib.URIRef(object_uri)
    
    s_types = [str(t) for t in engine.graph.objects(s_uri, RDF.type)]
    o_types = [str(t) for t in engine.graph.objects(o_uri, RDF.type)]
    
    if not s_types or not o_types:
        return "Error: Both subject and object must have defined types (use create_entity first)."

    # 1. Advanced Candidate Discovery (Direct, Facet, Inverse)
    candidates = engine.get_candidate_relations(s_types, o_types)
    
    if not candidates:
        return f"No valid ObjectProperties found between {s_types} and {o_types}, even via facets."

    # 2. Refine query with context
    combined_query = query
    if context:
        combined_query = f"{query} {context}"

    # 3. Rank candidates by semantic similarity
    candidate_uris = [c['uri'] for c in candidates]
    name_embs = [engine.properties[u]['name_embedding'] for u in candidate_uris]
    comment_embs = [engine.properties[u]['comment_embedding'] for u in candidate_uris]
    sim_scores = engine.weighted_similarity(combined_query, name_embs, comment_embs)
    
    # Apply score adjustments based on path type
    final_results = []
    for score, cand in zip(sim_scores, candidates):
        if cand['path_type'] == 'direct':
            score += DIRECT_RELATION_BOOST
        elif cand['path_type'] == 'inverse':
            score -= INVERSE_RELATION_PENALTY
        final_results.append((score, cand))
    
    results = sorted(final_results, key=lambda x: x[0], reverse=True)[:5]
    
    res = [f"Recommended relations: {subject_uri} -> {object_uri}"]
    res.append(f"   Intent/Context: '{combined_query}'")
    res.append("-" * 50)

    for i, (score, cand) in enumerate(results):
        uri = cand['uri']
        info = engine.properties[uri]
        res.append(f"{i+1}. {info['name']} ({uri}) [Score: {score:.4f}]")
        res.append(f"   Description: {info['comment'] or 'No description'}")
        
        # Path details and Tips
        if cand['path_type'] == 'facet':
            facet_name = cand['bridge'].split('/')[-1].split('#')[-1]
            res.append(f"   💡 TIP: This is a Facet-based connection via [{facet_name}].")
            res.append(f"           Use `attach_component` for this facet first.")
        elif cand['path_type'] == 'inverse':
            res.append(f"   💡 TIP: This is an INVERSE relation (defined as {o_types} -> {s_types}).")
            res.append(f"           Consider if the direction in your data should be reversed.")
        
        res.append("")
        
    return "\n".join(res)


@mcp.tool(structured_output=True)
def recommend_property(
    subject_type_uri: str,
    object_type_uri: str,
    predicate: str,
    context: str = None,
) -> RecommendPropertyResponse:
    """
    Recommends ontology properties for a subject/object type pair.
    It considers four directions:
    1. subject(entity) -> object(entity) via ObjectProperty
    2. object(entity) -> subject(entity) via ObjectProperty
    3. subject(entity) -> literal via DatatypeProperty
    4. object(entity) -> literal via DatatypeProperty

    The output is structured JSON with up to 3 candidates.
    """
    if not engine.has_class(subject_type_uri) or not engine.has_class(object_type_uri):
        return RecommendPropertyResponse(isSuccess=True, result=[])

    candidates = _build_recommend_property_candidates(subject_type_uri, object_type_uri)
    if not candidates:
        return RecommendPropertyResponse(isSuccess=True, result=[])

    combined_query = _combine_query_and_context(predicate, context)
    ranked_candidates = _rank_recommend_property_candidates(candidates, combined_query)
    if not ranked_candidates:
        return RecommendPropertyResponse(isSuccess=True, result=[])

    if not PROPERTY_RECOMMENDER_MODEL:
        logging.warning("recommend_property called without PROPERTY_RECOMMENDER_MODEL configured.")
        return RecommendPropertyResponse(isSuccess=False, result=[])

    messages = _build_recommend_property_messages(
        subject_type_uri,
        object_type_uri,
        predicate,
        context,
        ranked_candidates,
    )

    retry_messages = list(messages)
    last_error = "Unknown LLM failure."
    for _ in range(PROPERTY_RECOMMENDER_RETRY_LIMIT):
        raw_response = ""
        try:
            raw_response = _call_property_recommender_llm(retry_messages)
            parsed = json.loads(_extract_json_array(raw_response))
            validated = _validate_recommend_property_result(parsed, ranked_candidates)
            return RecommendPropertyResponse(isSuccess=True, result=validated)
        except Exception as exc:
            last_error = str(exc)
            retry_messages.append({"role": "assistant", "content": raw_response or "<empty response>"})
            retry_messages.append({
                "role": "user",
                "content": (
                    "Your previous answer was invalid.\n"
                    f"Reason: {last_error}\n"
                    "Return only a valid JSON array of up to 3 items with the exact keys "
                    "isReverse, isDataProperty, propertyURI, confidence."
                ),
            })

    logging.warning("recommend_property failed after retries: %s", last_error)
    return RecommendPropertyResponse(isSuccess=False, result=[])

@mcp.tool()
def get_ontology_summary() -> str:
    """
    Returns a high-level summary of the loaded ontology.
    Use this first to understand what kind of data the ontology can represent.
    """
    class_count = len(engine.classes)
    prop_count = len(engine.properties)
    shape_count = len(engine.shapes)
    
    # Extract unique namespaces/prefixes if possible
    namespaces = set()
    for uri in list(engine.classes.keys())[:100]: # Sample first 100
        if '#' in uri: namespaces.add(uri.split('#')[0] + '#')
        elif '/' in uri: namespaces.add('/'.join(uri.split('/')[:-1]) + '/')
        
    ns_list = "\n".join([f"- {ns}" for ns in sorted(list(namespaces))[:10]])
    
    return f"""
Ontology Summary:
- Total Classes: {class_count}
- Total Properties: {prop_count}
- SHACL Shapes: {shape_count}

Top Namespaces:
{ns_list}

Action Recommended: Use `list_root_classes` to begin navigation or `search_classes` for specific concepts.
"""

@mcp.tool()
def list_available_facets(class_uri: str) -> str:
    """
    Lists all Facets/Components applicable to a given class.
    Automatically ranks facets by semantic similarity to the target class.
    """
    if class_uri not in engine.classes: return "Error: Class not found."
    
    target_info = engine.classes[class_uri]
    
    # Collect all facets
    facet_candidates = []
    for uri, info in engine.classes.items():
        if "facet" in info['name'].lower() or "component" in info['name'].lower():
            facet_candidates.append((uri, info))
            
    if not facet_candidates: return "No specific facets found in this ontology."
    
    # Rank facets by weighted semantic similarity
    class_text = f"{target_info['name']} {target_info['comment']}"
    name_embeddings = [info['name_embedding'] for uri, info in facet_candidates]
    comment_embeddings = [info['comment_embedding'] for uri, info in facet_candidates]
    sim_scores = engine.weighted_similarity(class_text, name_embeddings, comment_embeddings)
    
    ranked_facets = []
    for i, (uri, info) in enumerate(facet_candidates):
        ranked_facets.append((sim_scores[i], uri, info))
        
    # Sort by similarity
    ranked_facets.sort(key=lambda x: x[0], reverse=True)
    
    res = [f"Recommended facets for {target_info['name']} (ranked by relevance):"]
    for score, uri, info in ranked_facets[:10]: # Top 10 for clarity
        res.append(f"- {info['name']} ({uri}) [Relevance: {score:.4f}]: {info['comment']}")
        
    return "\n".join(res)

@mcp.tool()
def get_class_details(class_uri: str) -> str:
    """Retrieves schema, hierarchy and connectivity guidance for a class."""
    if class_uri not in engine.classes: return "Error: Class not found."
    
    info = engine.classes[class_uri]
    supers = engine.get_transitive_superclasses(class_uri)
    
    res = [f"Class: {info['name']}", f"Comment: {info['comment']}", f"Hierarchy: {' -> '.join(supers[::-1] + [class_uri])}"]
    
    res.append("\n[Properties & Usage Guidance]")
    all_targets = [class_uri] + supers
    for target in all_targets:
        if target in engine.shapes:
            for s in engine.shapes[target]:
                p_info = engine.properties.get(s['path'], {})
                p_type = p_info.get('type', "Property")
                p_range = p_info.get('range', [])
                
                line = f"- {p_info.get('name', s['path'])} ({s['path']})\n  - Type: {p_type}"
                if p_type == "ObjectProperty":
                    line += f" (MUST link to an Entity of type: {', '.join(p_range) or 'Any'})"
                else:
                    line += f" (Literal value)"
                
                if s['minCount']: line += f" | ⚠️ REQUIRED"
                res.append(line)
    
    return "\n".join(res)

# --- [Entity Manipulation Tools] ---

@mcp.tool()
def create_entity(entity_id: str, class_uris: list[str]) -> str:
    """Creates a multi-typed entity. Base classes must exist in the loaded ontology."""
    safe_id = urllib.parse.quote(str(entity_id).replace(" ", "_"))
    entity_uri = rdflib.URIRef(engine.instance_base_uri + safe_id)

    # Detect existing entity
    existing_types = {str(t) for t in engine.graph.objects(entity_uri, RDF.type)}
    is_existing = len(existing_types) > 0

    STANDARD_NS = (
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://www.w3.org/2000/01/rdf-schema#",
        "http://www.w3.org/2002/07/owl#",
    )

    added_uris = []
    skipped_duplicate = []
    invalid_uris = []

    for c_uri in class_uris:
        c_uri_str = str(c_uri)
        if c_uri_str in engine.classes or any(c_uri_str.startswith(ns) for ns in STANDARD_NS):
            if c_uri_str in existing_types:
                skipped_duplicate.append(c_uri_str)
            else:
                engine.graph.add((entity_uri, RDF.type, rdflib.URIRef(c_uri_str)))
                added_uris.append(c_uri_str)
        else:
            invalid_uris.append(c_uri_str)

    lines = []
    if is_existing:
        existing_short = [t.split('/')[-1].split('#')[-1] for t in existing_types]
        lines.append(f"WARNING: Entity '{entity_uri}' already exists (types: {existing_short}).")
        if added_uris:
            lines.append(f"  Added new types: {added_uris}")
        if skipped_duplicate:
            lines.append(f"  Skipped (already assigned): {skipped_duplicate}")
    else:
        lines.append(f"Created: {entity_uri} (Original ID: {entity_id})")

    if invalid_uris:
        lines.append(f"WARNING: Classes not found in ontology and skipped: {invalid_uris}")

    return "\n".join(lines)

@mcp.tool()
def remove_entity(entity_uri: str) -> str:
    """Deletes an entity and all its associated links (incoming and outgoing)."""
    subj = rdflib.URIRef(entity_uri)
    engine.graph.remove((subj, None, None)) # Outgoing
    engine.graph.remove((None, None, subj)) # Incoming
    return f"Success: Removed {entity_uri} from the graph."

@mcp.tool()
def reset_graph() -> str:
    """Resets the knowledge graph (removes all instances) but keeps the ontology schema."""
    engine.graph = rdflib.Graph()
    engine.load_ontology() # Re-load schema
    return "Success: Knowledge graph has been reset to base ontology."


@mcp.tool()
def set_property(entity_uri: str, property_uri: str, value: str) -> str:

    """
    Sets a property value. The property_uri MUST exist in the loaded ontology
    or be from standard RDF/RDFS/OWL/SHACL namespaces.
    - value: The literal value OR the URI of another entity (must start with 'http').
    """
    if str(property_uri) not in engine.properties:
        # Check standard namespaces as well
        if not any(str(property_uri).startswith(ns) for ns in ["http://www.w3.org/1999/02/22-rdf-syntax-ns#", "http://www.w3.org/2000/01/rdf-schema#", "http://www.w3.org/2002/07/owl#", "http://www.w3.org/ns/shacl#"]):
             return f"ERROR: The property URI '{property_uri}' does not exist in the current ontology or standard namespaces."

    subj = rdflib.URIRef(entity_uri)
    pred = rdflib.URIRef(property_uri)
    
    # Auto-detect URI
    if str(value).startswith("http"):
        obj = rdflib.URIRef(value)
    else:
        obj = rdflib.Literal(value)
        
    engine.graph.add((subj, pred, obj))
    return f"Success: {entity_uri} --({property_uri})--> {value}"

@mcp.tool()
def attach_component(entity_uri: str, component_class: str, connection_prop: str, attributes: dict) -> str:

    """
    Groups metadata into a Facet/Component (e.g. FileFacet, NetworkConnectionFacet).
    - attributes: A dictionary of {property_uri: value}.
    - Note: Values starting with 'http' are automatically treated as URIs.
    """
    subj = rdflib.URIRef(entity_uri)
    # Safer URI cleaning
    cls_name = component_class.split('/')[-1].split('#')[-1]
    comp_uri = rdflib.URIRef(f"{entity_uri}_comp_{cls_name}_{uuid.uuid4().hex[:8]}")
    engine.graph.add((comp_uri, RDF.type, rdflib.URIRef(component_class)))
    engine.graph.add((subj, rdflib.URIRef(connection_prop), comp_uri))
    
    for p_uri, val in attributes.items():
        if str(val).startswith("http"):
            engine.graph.add((comp_uri, rdflib.URIRef(p_uri), rdflib.URIRef(val)))
        else:
            engine.graph.add((comp_uri, rdflib.URIRef(p_uri), rdflib.Literal(val)))
            
    return f"Success: Attached {component_class} to {entity_uri}. Component URI: {comp_uri}"

@mcp.tool()
def visualize_graph(verbose: bool = False) -> str:
    """
    Returns a text-based (ASCII tree) visualization of the current session's graph.
    - verbose: If True, displays full URIs for classes, properties, and entities.
    """
    if not engine.graph:
        return "The graph is currently empty."

    # Identify instance subjects
    subjects = set()
    for s in engine.graph.subjects():
        if str(s).startswith(engine.instance_base_uri):
            subjects.add(s)

    if not subjects:
        return "No instances found in the graph."

    mode_text = "(Full URIs)" if verbose else "(Names Only)"
    output = [f"### Current Knowledge Graph Preview {mode_text} ###\n"]
    
    # Simple grouping by type or subject
    for subj in sorted(subjects):
        subj_label = str(subj) if verbose else str(subj).replace(engine.instance_base_uri, "")
        types = [str(t) if verbose else str(t).split('/')[-1].split('#')[-1] for t in engine.graph.objects(subj, RDF.type)]
        output.append(f"● {subj_label} [{', '.join(types)}]")
        
        # Outgoing properties
        for p, o in engine.graph.predicate_objects(subj):
            if p == RDF.type: continue
            
            p_label = str(p) if verbose else str(p).split('/')[-1].split('#')[-1]
            if str(o).startswith(engine.instance_base_uri):
                o_label = str(o) if verbose else str(o).replace(engine.instance_base_uri, "")
                output.append(f"  └── {p_label} ➔ {o_label}")
            else:
                o_val = str(o)
                output.append(f"  └── {p_label}: \"{o_val}\"")
        output.append("")

    return "\n".join(output)

@mcp.tool()
def get_graph_data() -> str:
    """Returns the current knowledge graph as a JSON string containing nodes and edges."""
    import json
    nodes_map = {}
    edges = []
    
    # Base URI check
    def is_instance(uri):
        return str(uri).startswith(engine.instance_base_uri)

    # 1. First pass: Collect all instances as subjects
    for s in engine.graph.subjects():
        if is_instance(s):
            s_uri = str(s)
            if s_uri not in nodes_map:
                nodes_map[s_uri] = {
                    "id": s_uri,
                    "label": s_uri.replace(engine.instance_base_uri, ""),
                    "types": []
                }
            
            # Attributes and types
            for p, o in engine.graph.predicate_objects(s):
                p_str = str(p)
                o_str = str(o)
                
                if p == RDF.type:
                    short_type = o_str.split('/')[-1].split('#')[-1]
                    if short_type not in nodes_map[s_uri]["types"]:
                        nodes_map[s_uri]["types"].append(short_type)
                elif is_instance(o):
                    # It's an edge to another instance
                    edges.append({
                        "from": s_uri,
                        "to": o_str,
                        "label": p_str.split('/')[-1].split('#')[-1]
                    })
                    # Ensure the target node exists in nodes_map too
                    if o_str not in nodes_map:
                        nodes_map[o_str] = {
                            "id": o_str,
                            "label": o_str.replace(engine.instance_base_uri, ""),
                            "types": []
                        }
                else:
                    # It's an attribute
                    key = p_str.split('/')[-1].split('#')[-1]
                    # Don't overwrite if multiple, just append or cap
                    nodes_map[s_uri][key] = o_str

    return json.dumps({"nodes": list(nodes_map.values()), "edges": edges}, indent=2)



@mcp.tool()
def get_raw_triplets() -> str:
    """Debug tool: returns ALL triplets in the graph as text."""
    res = []
    for s, p, o in engine.graph:
        res.append(f"S: {s} | P: {p} | O: {o}")
    return "\n".join(res) if res else "Graph is empty."


@mcp.tool()
def prune_islands(min_size: int = 1) -> str:
    """
    Removes disconnected 'islands' (small clusters of entities) from the graph.
    - min_size: Islands with total entity count <= min_size will be removed.
    - Helps clean up semantic noise/side-talk that doesn't connect to the core analysis.
    """
    import networkx as nx
    G = nx.Graph()
    
    def is_instance(uri):
        return str(uri).startswith(engine.instance_base_uri)
        
    instance_nodes = set()
    # Find all instances
    for s, p, o in engine.graph:
        if is_instance(s):
            instance_nodes.add(s)
            if is_instance(o):
                G.add_edge(s, o)
                instance_nodes.add(o)
        elif is_instance(o):
            instance_nodes.add(o)
            
    # Add isolated nodes to the graph
    for node in instance_nodes:
        if node not in G:
            G.add_node(node)
            
    # Identify components to prune
    components = list(nx.connected_components(G))
    to_remove = set()
    for comp in components:
        if len(comp) <= min_size:
            to_remove.update(comp)
            
    # Delete triples associated with pruned entities
    triples_removed = 0
    for entity in to_remove:
        # Outgoing
        for p, o in list(engine.graph.predicate_objects(entity)):
            # If o is a facet/component (we hash them in attach_component), we should follow it?
            # For now, let's keep it simple: any triple where subject or object is a pruned instance
            engine.graph.remove((entity, p, o))
            triples_removed += 1
        # Incoming
        for s, p in list(engine.graph.subject_predicates(entity)):
            engine.graph.remove((s, p, entity))
            triples_removed += 1
            
    return f"Success: Pruned {len(to_remove)} entities across {len([c for c in components if len(c) <= min_size])} islands. Total triples removed: {triples_removed}."


@mcp.tool()
def validate_entity(entity_uri: str) -> str:
    """Validates entity against SHACL constraints. Returns structured JSON."""
    subj = rdflib.URIRef(entity_uri)
    types = [str(t) for t in engine.graph.objects(subj, RDF.type)]

    if not types:
        result = {
            "valid": False,
            "entity": entity_uri,
            "errors": [{
                "type": "no_type",
                "message": "Entity has no rdf:type assigned.",
                "fix": f"Use create_entity('{entity_uri.split('/')[-1]}', ['<class_uri>']) to assign a class."
            }]
        }
        return json.dumps(result, indent=2)

    all_targets = []
    for t in types:
        all_targets.extend([t] + engine.get_transitive_superclasses(t))

    errors = []
    for target in set(all_targets):
        if target not in engine.shapes:
            continue
        target_short = target.split('/')[-1].split('#')[-1]
        for shape in engine.shapes[target]:
            values = list(engine.graph.objects(subj, rdflib.URIRef(shape['path'])))
            min_count = int(shape['minCount']) if shape['minCount'] is not None else None
            max_count = int(shape['maxCount']) if shape['maxCount'] is not None else None

            if min_count is not None and len(values) < min_count:
                p_info = engine.properties.get(shape['path'], {})
                p_name = p_info.get('name', shape['path'].split('/')[-1].split('#')[-1])
                p_type = p_info.get('type', 'Property')
                p_range = p_info.get('range', [])
                range_short = [r.split('/')[-1].split('#')[-1] for r in p_range]

                if p_type == "ObjectProperty":
                    fix = (f"Use set_property('{entity_uri}', '{shape['path']}', '<target_entity_uri>') "
                           f"to link an entity of type: {range_short or 'Any'}.")
                else:
                    fix = f"Use set_property('{entity_uri}', '{shape['path']}', '<value>') to set the missing literal."

                errors.append({
                    "type": "missing_required_property",
                    "property_uri": shape['path'],
                    "property_name": p_name,
                    "property_type": p_type,
                    "expected_range": range_short,
                    "required_by": target_short,
                    "current_count": len(values),
                    "min_required": min_count,
                    "fix": fix,
                })

            if max_count is not None and len(values) > max_count:
                p_info = engine.properties.get(shape['path'], {})
                p_name = p_info.get('name', shape['path'].split('/')[-1].split('#')[-1])
                errors.append({
                    "type": "excess_values",
                    "property_uri": shape['path'],
                    "property_name": p_name,
                    "required_by": target_short,
                    "current_count": len(values),
                    "max_allowed": max_count,
                    "fix": f"Remove extra values for '{shape['path']}' until only {max_count} remain.",
                })

    if not errors:
        return json.dumps({"valid": True, "entity": entity_uri, "message": "All SHACL constraints satisfied."})
    return json.dumps({"valid": False, "entity": entity_uri, "error_count": len(errors), "errors": errors}, indent=2)

@mcp.tool()
def export_graph(filename: str = "knowledge_graph.ttl") -> str:
    """Exports all instance data to a Turtle file. Fails if invalid URIs exist."""
    output = rdflib.Graph()
    for s, p, o in engine.graph:
        if str(s).startswith(engine.instance_base_uri):
            output.add((s, p, o))
    
    try:
        output.serialize(destination=filename, format="turtle")
        return f"Successfully exported to {filename}"
    except Exception as e:
        return f"Export Error: {str(e)}. This usually means an invalid URI was created. Consider using `reset_graph` and avoiding special characters in IDs."

def run():
    mcp.run()
