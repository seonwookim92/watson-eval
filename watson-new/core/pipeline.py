"""OntologyExtractor pipeline.

Orchestrated by LangGraph as a sequential graph:
  Pre-processing -> Chunking -> Paraphrasing -> Triplet Extraction ->
  Entity Extraction -> IoC Detection -> Triplet Type Matching ->
  Internal Entity Resolution -> Existing Entity Resolution -> Data Insert
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote, urlparse
from uuid import uuid4

import requests
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from .clients import EmbeddingClient, LLMClient, MCPStdioClient, build_embedding_client
from .config import load_config
from .prompts import (
    class_resolution_agent_prompt,
    chunk_prompt,
    data_property_check_prompt,
    data_property_resolution_agent_prompt,
    entity_extraction_prompt,
    entity_inventory_from_triplets_prompt,
    entity_pre_classification_prompt,
    entity_resolution_prompt,
    ioc_prompt,
    object_property_resolution_agent_prompt,
    paraphrase_stage1_prompt,
    paraphrase_stage1_retry_prompt,
    paraphrase_stage1_verify_prompt,
    paraphrase_stage2_prompt,
    paraphrase_stage2_retry_prompt,
    paraphrase_stage2_verify_prompt,
    paraphrase_stage3_prompt,
    paraphrase_stage3_retry_prompt,
    paraphrase_stage3_verify_prompt,
    paraphrase_stage4_prompt,
    paraphrase_stage4_retry_prompt,
    paraphrase_stage4_verify_prompt,
    predicate_candidate_judge_prompt,
    predicate_match_select_prompt,
    predicate_query_expansion_prompt,
    qualifier_node_worthiness_prompt,
    span_normalization_prompt,
    triplet_prompt,
    type_match_select_prompt,
)
from .storage import VectorStore
from .utils import (
    clean_text,
    is_url,
    normalize_plain_text,
    safe_filename,
    setup_logger,
    to_iso_now,
)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    error: Optional[str]


EVIDENCE_KIND_WEIGHTS: Dict[str, int] = {
    "explicit_definition": 5,
    "apposition": 4,
    "name_pattern": 3,
    "entity_description": 2,
    "relation_context": 1,
}


class OntologyConstraintChecker:
    @staticmethod
    def _detect_format(schema_path: Path) -> Optional[str]:
        return {
            ".ttl": "ttl",
            ".owl": "xml",
            ".rdf": "xml",
        }.get(schema_path.suffix.lower())

    def _collect_ontology_files(self, schema_path: Path) -> List[Tuple[Path, str]]:
        files: List[Tuple[Path, str]] = []
        if schema_path.is_dir():
            for pattern, fmt in (("**/*.ttl", "ttl"), ("**/*.owl", "xml"), ("**/*.rdf", "xml")):
                for file_path in sorted(schema_path.glob(pattern)):
                    files.append((file_path, fmt))
            return files

        fmt = self._detect_format(schema_path)
        if fmt:
            files.append((schema_path, fmt))

        # STIX-style master ontologies often import many sibling OWL files via a local XML catalog.
        # Parse the containing ontology directory as well so class/property counts reflect the full schema.
        parent = schema_path.parent
        if (parent / "catalog-v001.xml").exists():
            seen = {item[0].resolve() for item in files}
            for pattern, fmt in (("**/*.ttl", "ttl"), ("**/*.owl", "xml"), ("**/*.rdf", "xml")):
                for file_path in sorted(parent.glob(pattern)):
                    resolved = file_path.resolve()
                    if resolved in seen:
                        continue
                    files.append((file_path, fmt))
                    seen.add(resolved)
        return files

    def __init__(self, ontology_schema_file: str, logger: logging.Logger) -> None:
        self.logger = logger
        self.available = False
        self.class_parents: Dict[str, set[str]] = {}
        self.object_properties: Dict[str, Dict[str, set[str]]] = {}
        self.data_properties: Dict[str, Dict[str, set[str]]] = {}
        try:
            from rdflib import Graph, RDF, RDFS, OWL, URIRef  # type: ignore

            graph = Graph()
            schema_path = Path(ontology_schema_file).resolve()
            files = self._collect_ontology_files(schema_path)
            if not files:
                raise FileNotFoundError(f"No ontology files found under schema path: {schema_path}")
            for file_path, fmt in files:
                graph.parse(str(file_path), format=fmt)
            self._rdf = {"RDF": RDF, "RDFS": RDFS, "OWL": OWL, "URIRef": URIRef}

            named_classes = {
                cls for cls in
                set(graph.subjects(RDF.type, OWL.Class)) | set(graph.subjects(RDF.type, RDFS.Class))
                if isinstance(cls, URIRef)
            }
            for cls in named_classes:
                cls_uri = str(cls)
                self.class_parents.setdefault(cls_uri, set())
                for parent in graph.objects(cls, RDFS.subClassOf):
                    if isinstance(parent, URIRef):
                        self.class_parents[cls_uri].add(str(parent))

            for prop in graph.subjects(RDF.type, OWL.ObjectProperty):
                prop_uri = str(prop)
                self.object_properties[prop_uri] = {
                    "domain": {str(o) for o in graph.objects(prop, RDFS.domain) if isinstance(o, URIRef)},
                    "range": {str(o) for o in graph.objects(prop, RDFS.range) if isinstance(o, URIRef)},
                }

            for prop in graph.subjects(RDF.type, OWL.DatatypeProperty):
                prop_uri = str(prop)
                self.data_properties[prop_uri] = {
                    "domain": {str(o) for o in graph.objects(prop, RDFS.domain) if isinstance(o, URIRef)},
                    "range": set(),
                }

            self.available = True
            logger.info(
                "[OntologyChecker] Loaded schema at init. classes=%s object_properties=%s data_properties=%s",
                len(self.class_parents),
                len(self.object_properties),
                len(self.data_properties),
            )
        except Exception as e:
            logger.warning("[OntologyChecker] Disabled - %s", e)

    def _ancestor_closure(self, class_uri: str) -> set[str]:
        seen: set[str] = set()
        stack = [class_uri]
        while stack:
            current = stack.pop()
            if not current or current in seen:
                continue
            seen.add(current)
            stack.extend(self.class_parents.get(current, set()) - seen)
        return seen

    def _class_matches(self, class_uri: str, allowed: set[str]) -> bool:
        if not allowed:
            return True
        if not class_uri:
            return False
        closure = self._ancestor_closure(class_uri)
        return any(candidate in allowed for candidate in closure)

    def validate_object_property(self, property_uri: str, subject_class_uri: str, object_class_uri: str) -> bool:
        if not self.available:
            return bool(property_uri and subject_class_uri and object_class_uri)
        info = self.object_properties.get(property_uri)
        if not info:
            return False
        return self._class_matches(subject_class_uri, info["domain"]) and self._class_matches(object_class_uri, info["range"])

    def validate_data_property(self, property_uri: str, subject_class_uri: str) -> bool:
        if not self.available:
            return bool(property_uri and subject_class_uri)
        info = self.data_properties.get(property_uri)
        if not info:
            return False
        return self._class_matches(subject_class_uri, info["domain"])

    def find_object_properties(
        self, subject_class_uri: str, object_class_uri: str
    ) -> List[str]:
        if not (self.available and subject_class_uri and object_class_uri):
            return []
        matches: List[str] = []
        for prop_uri in self.object_properties:
            if self.validate_object_property(prop_uri, subject_class_uri, object_class_uri):
                matches.append(prop_uri)
        return matches

    def has_class_uri(self, class_uri: str) -> bool:
        return bool(self.available and class_uri in self.class_parents)

    def has_object_property_uri(self, property_uri: str) -> bool:
        return bool(self.available and property_uri in self.object_properties)

    def has_data_property_uri(self, property_uri: str) -> bool:
        return bool(self.available and property_uri in self.data_properties)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class OntologyExtractorPipeline:
    def __init__(
        self,
        input_source: str,
        ontology_schema_file: str,
        config_path: str = "config.json",
    ) -> None:
        self.input_source = input_source
        self.ontology_schema_file = ontology_schema_file
        self.config = load_config(Path(config_path))

        self.run_root = Path.cwd() / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.intermediate = self.run_root / "intermediate"
        self.preprocess_dir = self.intermediate / "preprocess"
        self.chunking_dir = self.intermediate / "chunking"
        self.paraphrase_dir = self.intermediate / "paraphrase"

        # Create directory structure
        for d in (self.run_root, self.intermediate, self.preprocess_dir,
                  self.chunking_dir, self.paraphrase_dir):
            d.mkdir(parents=True, exist_ok=True)

        log_path = self.run_root / "run.log"
        self.logger = setup_logger(log_path)
        self.trace_dir = self.run_root / "traces"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.mcp_trace_path = self.trace_dir / "mcp_agent_full.jsonl"

        self.llm = LLMClient(
            self.config["llm"]["base_url"],
            self.config["llm"]["model"],
            max_tokens=int(self.config["llm"].get("max_tokens", 4096)),
            thinking=bool(self.config["llm"].get("thinking", False)),
            logger=self.logger,
            tracer=self._trace_llm_event,
        )
        self.embedding = build_embedding_client(
            mode=str(self.config["embedding"].get("mode", "local")),
            base_url=str(self.config["embedding"].get("base_url", "")),
            model=self.config["embedding"]["model"],
            truncate_prompt_tokens=self.config["embedding"]["truncate_prompt_tokens"],
            api_key=self.config["embedding"].get("api_key", ""),
        )
        self.ontology_checker = OntologyConstraintChecker(self.ontology_schema_file, self.logger)
        self.vector_store = VectorStore(
            self.config["zvec"]["store_path"],
            self.config["zvec"],
            self.logger,
            self.embedding,
        )

        # Pipeline state
        self.preprocess_file: Optional[Path] = None
        self.chunk_files: List[Path] = []
        self.paraphrase_files: List[Path] = []
        self.entities: List[Dict[str, str]] = []
        self.entities_by_chunk: Dict[str, List[Dict[str, str]]] = {}
        self.entity_memory: Dict[str, Dict[str, Any]] = {}
        self.triplets: List[Dict[str, Any]] = []
        self.all_typed_triplets: List[Dict[str, Any]] = []
        self.typed_triplets: List[Dict[str, Any]] = []
        self.mcp_call_count: Dict[str, int] = {"type_matching": 0, "property_matching": 0}
        self.report_root: Dict[str, str] = {"name": "", "class_uri": "", "class_name": ""}

        # Node 7 caches: avoid redundant MCP calls for identical entities/predicates
        self._entity_class_cache: Dict[str, Tuple[str, str]] = {}
        self._ioc_type_class_cache: Dict[str, Tuple[str, str]] = {}
        self._literal_check_cache: Dict[Tuple[str, str], Tuple[bool, str]] = {}
        self._predicate_uri_cache: Dict[Tuple[str, str, str, str, str], Tuple[str, bool, bool]] = {}
        self._predicate_query_cache: Dict[Tuple[str, str, str], List[str]] = {}
        self._untyped_predicate_cache: Dict[str, str] = {}
        self._chunk_text_cache: Dict[str, str] = {}

        # iocsearcher: lazy-init once, reused across all _ioc_matches calls
        self._ioc_searcher: Any = None
        self._ioc_searcher_loaded: bool = False

        # Locks for thread-safe cache access and shared counter updates
        self._cache_lock = threading.Lock()
        self._mcp_count_lock = threading.Lock()
        self._progress_lock = threading.Lock()
        self._trace_lock = threading.Lock()
        self._type_matching_totals: Dict[str, int] = {
            "triplets": 0,
            "entity_slots": 0,
            "relationships": 0,
        }
        self._type_matching_progress: Dict[str, int] = {
            "triplets": 0,
            "entity_slots": 0,
            "relationships": 0,
        }

        # Derive source filename
        self.source_filename = safe_filename(os.path.basename(input_source))
        if is_url(input_source):
            parsed = urlparse(input_source)
            candidate = os.path.basename(unquote(parsed.path))
            self.source_filename = safe_filename(candidate) if candidate else safe_filename(
                parsed.netloc.replace(".", "_")
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, level: int, message: str, *args: Any) -> None:
        self.logger.log(level, message, *args)

    @staticmethod
    def _truncate_for_log(value: Any, limit: int = 4000) -> str:
        text = str(value)
        if len(text) <= limit:
            return text
        return f"{text[:limit]}... [truncated {len(text) - limit} chars]"

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False)
        with self._trace_lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")

    def _sanitize_trace_value(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): self._sanitize_trace_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize_trace_value(item) for item in value]
        return str(value)

    def _trace_event(self, event: str, **payload: Any) -> None:
        record = {
            "timestamp": to_iso_now(),
            "run_id": self.run_root.name,
            "event": event,
            **{key: self._sanitize_trace_value(value) for key, value in payload.items()},
        }
        self._append_jsonl(self.mcp_trace_path, record)

    def _trace_llm_event(self, payload: Dict[str, Any]) -> None:
        event = str(payload.get("event", "llm_exchange"))
        trace_payload = dict(payload)
        trace_payload.pop("event", None)
        self._trace_event(event, **trace_payload)

    def _read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")

    def _split_sentences(self, text: str) -> List[str]:
        try:
            import nltk
            return [s for s in nltk.sent_tokenize(text) if s.strip()]
        except ImportError:
            pass
        except Exception as e:
            self._log(logging.WARNING, "NLTK tokenization failed (%s), using regex fallback", e)
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _reset_type_matching_progress(self) -> None:
        with self._progress_lock:
            self._type_matching_totals = {
                "triplets": len(self.triplets),
                "entity_slots": len(self.triplets) * 2,
                "relationships": len(self.triplets),
            }
            self._type_matching_progress = {
                "triplets": 0,
                "entity_slots": 0,
                "relationships": 0,
            }

    def _advance_type_matching_progress(self) -> Dict[str, int]:
        with self._progress_lock:
            self._type_matching_progress["triplets"] += 1
            self._type_matching_progress["entity_slots"] += 2
            self._type_matching_progress["relationships"] += 1
            return {
                "triplet_index": self._type_matching_progress["triplets"],
                "triplet_total": self._type_matching_totals["triplets"],
                "entity_index": self._type_matching_progress["entity_slots"],
                "entity_total": self._type_matching_totals["entity_slots"],
                "relationship_index": self._type_matching_progress["relationships"],
                "relationship_total": self._type_matching_totals["relationships"],
            }

    @staticmethod
    def _format_type_match_status(name: str, class_name: str, class_uri: str) -> str:
        if class_uri:
            display = class_name or class_uri.rsplit("#", 1)[-1]
            return f'"{name}" matched type "{display}" ({class_uri})'
        return f'"{name}" matching failed'

    @staticmethod
    def _format_relationship_match_status(predicate: str, predicate_uri: str) -> str:
        if predicate_uri:
            return f'"{predicate}" matched property "{predicate_uri}"'
        return f'"{predicate}" matching failed'

    def _log_type_matching_result(
        self,
        row: Dict[str, Any],
        typed_row: Dict[str, Any],
        progress: Dict[str, int],
    ) -> None:
        subject_status = self._format_type_match_status(
            typed_row["subject"],
            typed_row["subject_class_name"],
            typed_row["subject_class_uri"],
        )
        if typed_row["object_is_literal"]:
            if typed_row["predicate_uri"]:
                object_status = (
                    f'"{typed_row["object"]}" treated as literal with property '
                    f'"{typed_row["predicate_uri"]}"'
                )
            else:
                object_status = f'"{typed_row["object"]}" treated as literal but property matching failed'
        else:
            object_status = self._format_type_match_status(
                typed_row["object"],
                typed_row["object_class_name"],
                typed_row["object_class_uri"],
            )
        relationship_status = self._format_relationship_match_status(
            typed_row["predicate"],
            typed_row["predicate_uri"],
        )
        self._log(
            logging.INFO,
            (
                '[7][Triplet %s/%s][Entity %s/%s][Relationship %s/%s] '
                'id=%s | subject=%s | object=%s | relationship=%s'
            ),
            progress["triplet_index"],
            progress["triplet_total"],
            progress["entity_index"],
            progress["entity_total"],
            progress["relationship_index"],
            progress["relationship_total"],
            row["id"],
            subject_status,
            object_status,
            relationship_status,
        )

    # ------------------------------------------------------------------
    # Node 1: Pre-processing
    # ------------------------------------------------------------------

    def node_pre_processing(self) -> None:
        self._log(logging.INFO, "[1] Pre-processing start (%s)", to_iso_now())
        if not self._run_textitdown():
            self._log(logging.WARNING, "[1] Fallback to direct read for preprocess")
            if not self._fallback_preprocess():
                raise RuntimeError("Preprocessing failed")
        if not self.preprocess_file or not self.preprocess_file.exists():
            raise RuntimeError("Preprocessing output not found")
        normalized = normalize_plain_text(self._read_text(self.preprocess_file))
        self.preprocess_file.write_text(normalized, encoding="utf-8")
        self._log(logging.INFO, "[1] Pre-processing completed: %s", self.preprocess_file)

    def _run_textitdown(self) -> bool:
        script = Path("TextItDown/textitdown.py")
        if not script.exists():
            self._log(logging.ERROR, "[Pre-processing] TextItDown script not found: %s", script)
            return False
        before = set(self.preprocess_dir.glob("*"))
        output_file = self.preprocess_dir / f"{self.source_filename}.txt"
        command = [sys.executable, str(script), self.input_source, str(output_file)]
        if self.config.get("force_pdf_ocr", False):
            command.append("--force-pdf-ocr")
        self._log(logging.INFO, "[Pre-processing] Running TextItDown: %s", " ".join(command))
        textitdown_timeout = int(self.config.get("textitdown_timeout", 600))
        try:
            proc = subprocess.run(
                command,
                cwd=str(Path.cwd()),
                capture_output=True,
                text=True,
                timeout=textitdown_timeout,
            )
        except subprocess.TimeoutExpired:
            self._log(logging.ERROR, "[Pre-processing] TextItDown timed out after %ss", textitdown_timeout)
            return False
        if proc.returncode != 0:
            self._log(logging.ERROR, "[Pre-processing] TextItDown failed: %s", proc.stderr.strip())
            return False
        after = set(self.preprocess_dir.glob("*"))
        new_files = list(after - before)
        if not new_files:
            self._log(logging.ERROR, "[Pre-processing] TextItDown did not create output file")
            return False
        preferred = [p for p in new_files if p.name.startswith(self.source_filename)]
        self.preprocess_file = preferred[0] if preferred else new_files[0]
        return True

    def _fallback_preprocess(self) -> bool:
        if not self.preprocess_file:
            self.preprocess_file = self.preprocess_dir / f"{self.source_filename}.txt"
        if is_url(self.input_source):
            try:
                resp = requests.get(self.input_source, timeout=120)
                resp.raise_for_status()
                self.preprocess_file.write_text(resp.text, encoding="utf-8")
                return True
            except Exception as e:
                self._log(logging.ERROR, "[Pre-processing] URL fallback failed: %s", e)
                return False
        if os.path.isfile(self.input_source):
            shutil.copyfile(self.input_source, self.preprocess_file)
            return True
        self._log(logging.ERROR, "[Pre-processing] Unsupported source type: %s", self.input_source)
        return False

    # ------------------------------------------------------------------
    # Node 2: Chunking
    # ------------------------------------------------------------------

    def node_chunking(self) -> None:
        self._log(logging.INFO, "[2] Chunking start")
        self.chunk_files = []
        raw_text = self._read_text(self.preprocess_file)
        if not raw_text.strip():
            return
        method = str(self.config.get("chunking", {}).get("method", "semantic")).strip().lower()
        max_bytes = int(self.config["chunking"]["max_chunk_bytes"])
        retries = int(self.config["retry"].get("semantic_chunking", 3))
        if method == "semantic":
            chunks = self._build_semantic_chunks(raw_text, max_bytes, retries)
        else:
            chunks = self._build_chunks_by_paragraphs(raw_text, max_bytes)
        for chunk_index, chunk_text in enumerate(chunks):
            path = self.chunking_dir / f"chunk_{chunk_index:04d}.txt"
            path.write_text(chunk_text, encoding="utf-8")
            self.chunk_files.append(path)

        if not self.chunk_files:
            fallback = self.chunking_dir / "chunk_0000.txt"
            fallback.write_text(raw_text, encoding="utf-8")
            self.chunk_files.append(fallback)

        self._log(logging.INFO, "[2] Chunking completed. chunks=%s", len(self.chunk_files))

    @staticmethod
    def _byte_len(text: str) -> int:
        return len(text.encode("utf-8"))

    def _prefix_within_bytes(self, text: str, max_bytes: int) -> str:
        if self._byte_len(text) <= max_bytes:
            return text
        current = ""
        for char in text:
            candidate = f"{current}{char}"
            if current and self._byte_len(candidate) > max_bytes:
                break
            current = candidate
        return current

    @staticmethod
    def _looks_like_sentence_end(text: str) -> bool:
        return bool(re.search(r"[.!?][\"')\]]*\s*$", text.strip()))

    def _expand_partial_to_sentence(self, block: str, returned_text: str) -> str:
        candidate = clean_text(returned_text)
        if not candidate:
            return ""

        match_index = block.rfind(candidate)
        if match_index < 0:
            return ""
        if self._looks_like_sentence_end(candidate):
            return candidate

        sentence_end = match_index + len(candidate)
        while sentence_end < len(block):
            if block[sentence_end] in ".!?":
                sentence_end += 1
                while sentence_end < len(block) and block[sentence_end] in "\"')]} \n\t":
                    sentence_end += 1
                return block[match_index:sentence_end].strip()
            sentence_end += 1
        return ""

    def _semantic_chunk_end(self, block: str, retries: int) -> str:
        for attempt in range(1, retries + 1):
            try:
                parsed = self.llm.chat_json(chunk_prompt(block))
            except Exception as e:
                self._log(logging.WARNING, "[2] Semantic chunking attempt %s failed: %s", attempt, e)
                continue
            if not isinstance(parsed, dict):
                continue
            last_sentence = clean_text(parsed.get("last_sentence", ""))
            expanded = self._expand_partial_to_sentence(block, last_sentence)
            if expanded:
                return expanded
        return ""

    def _build_semantic_chunks(self, text: str, max_bytes: int, retries: int) -> List[str]:
        chunks: List[str] = []
        cursor = 0

        while cursor < len(text):
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor >= len(text):
                break

            remaining = text[cursor:]
            if self._byte_len(remaining) <= max_bytes:
                chunks.append(remaining.strip())
                break

            block = self._prefix_within_bytes(remaining, max_bytes)
            last_sentence = self._semantic_chunk_end(block, retries)
            if not last_sentence:
                self._log(logging.WARNING, "[2] Semantic chunking fallback triggered at offset=%s", cursor)
                chunks.extend(self._build_chunks_by_paragraphs(remaining, max_bytes))
                break

            cut = block.rfind(last_sentence)
            if cut < 0:
                self._log(logging.WARNING, "[2] Semantic chunking could not locate the chosen sentence at offset=%s", cursor)
                chunks.extend(self._build_chunks_by_paragraphs(remaining, max_bytes))
                break

            chunk_text = block[:cut + len(last_sentence)].strip()
            if not chunk_text:
                self._log(logging.WARNING, "[2] Semantic chunking produced an empty chunk at offset=%s", cursor)
                chunks.extend(self._build_chunks_by_paragraphs(remaining, max_bytes))
                break

            chunks.append(chunk_text)
            cursor += cut + len(last_sentence)

        return [chunk for chunk in chunks if chunk.strip()]

    def _build_chunks_by_paragraphs(self, text: str, max_bytes: int) -> List[str]:
        paragraphs = [line.strip() for line in text.split("\n") if line.strip()]
        chunks: List[str] = []
        current_parts: List[str] = []
        current_bytes = 0

        for paragraph in paragraphs:
            paragraph_units = self._split_oversized_paragraph(paragraph, max_bytes)
            for unit in paragraph_units:
                separator_bytes = 1 if current_parts else 0
                unit_bytes = self._byte_len(unit)
                if current_parts and current_bytes + separator_bytes + unit_bytes > max_bytes:
                    chunks.append("\n".join(current_parts))
                    current_parts = [unit]
                    current_bytes = unit_bytes
                    continue
                current_parts.append(unit)
                current_bytes += separator_bytes + unit_bytes

        if current_parts:
            chunks.append("\n".join(current_parts))
        return chunks

    def _split_oversized_paragraph(self, paragraph: str, max_bytes: int) -> List[str]:
        if self._byte_len(paragraph) <= max_bytes:
            return [paragraph]

        sentences = self._split_sentences(paragraph)
        if len(sentences) <= 1:
            return self._split_text_by_bytes(paragraph, max_bytes)

        units: List[str] = []
        current_sentences: List[str] = []
        current_bytes = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_bytes = self._byte_len(sentence)
            if sentence_bytes > max_bytes:
                if current_sentences:
                    units.append(" ".join(current_sentences))
                    current_sentences = []
                    current_bytes = 0
                units.extend(self._split_text_by_bytes(sentence, max_bytes))
                continue
            separator_bytes = 1 if current_sentences else 0
            if current_sentences and current_bytes + separator_bytes + sentence_bytes > max_bytes:
                units.append(" ".join(current_sentences))
                current_sentences = [sentence]
                current_bytes = sentence_bytes
                continue
            current_sentences.append(sentence)
            current_bytes += separator_bytes + sentence_bytes

        if current_sentences:
            units.append(" ".join(current_sentences))
        return units or [paragraph]

    def _split_text_by_bytes(self, text: str, max_bytes: int) -> List[str]:
        chunks: List[str] = []
        current = ""
        for char in text:
            candidate = f"{current}{char}"
            if current and self._byte_len(candidate) > max_bytes:
                chunks.append(current)
                current = char
                continue
            current = candidate
        if current:
            chunks.append(current)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    # ------------------------------------------------------------------
    # Node 3: Paraphrasing
    # ------------------------------------------------------------------

    _PRONOUN_PATTERN = re.compile(
        r"\b(he|she|it|they|his|her|its|their|him|them|this|that|these|those)\b",
        flags=re.IGNORECASE,
    )

    def _write_paraphrase_attempt(self, idx: int, stage: str, attempt: int, text: str) -> None:
        tried = self.paraphrase_dir / f"chunk_{idx:04d}_{stage}_try_{attempt:02d}.txt"
        tried.write_text(text, encoding="utf-8")

    def _annotate_pronouns(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        items: List[Dict[str, Any]] = []
        counter = 0

        def repl(match: re.Match[str]) -> str:
            nonlocal counter
            counter += 1
            pronoun = match.group(0)
            items.append({"id": counter, "pronoun": pronoun})
            return f"<{pronoun}:{counter}>"

        annotated = self._PRONOUN_PATTERN.sub(repl, text)
        return annotated, items

    @staticmethod
    def _pronoun_guide(items: List[Dict[str, Any]]) -> str:
        return "\n".join(f'- {item["id"]}: {item["pronoun"]}' for item in items) or "(none)"

    @staticmethod
    def _replacements_to_json(payload: Any) -> str:
        try:
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return str(payload)

    def _apply_pronoun_replacements(
        self,
        annotated: str,
        items: List[Dict[str, Any]],
        response: Any,
    ) -> Tuple[str, bool]:
        if not isinstance(response, dict):
            return annotated, False
        replacements = response.get("replacements")
        if not isinstance(replacements, list):
            return annotated, False
        replacement_map: Dict[int, str] = {}
        for entry in replacements:
            if not isinstance(entry, dict):
                continue
            try:
                pronoun_id = int(entry.get("id"))
            except Exception:
                continue
            replacement = clean_text(str(entry.get("replacement", "")))
            if replacement:
                replacement_map[pronoun_id] = replacement

        updated = annotated
        success = True
        for item in items:
            tag = f'<{item["pronoun"]}:{item["id"]}>'
            replacement = replacement_map.get(int(item["id"]))
            if not replacement:
                success = False
                continue
            updated = updated.replace(tag, replacement)
        if re.search(r"<[^>]+:\d+>", updated):
            success = False
        return updated, success

    def _run_stage1(self, idx: int, original: str, retry_limit: int) -> str:
        latest_safe = ""
        current = original
        issues = ""
        for attempt in range(1, retry_limit + 1):
            try:
                if attempt == 1:
                    current = self.llm.chat_text(paraphrase_stage1_prompt(original))
                else:
                    current = self.llm.chat_text(paraphrase_stage1_retry_prompt(original, current, issues))
                self._write_paraphrase_attempt(idx, "stage1", attempt, current)
                check = self.llm.chat_json(paraphrase_stage1_verify_prompt(original, current))
            except Exception as e:
                self._log(logging.WARNING, "[3-1] Chunk %s attempt %s LLM error: %s", idx, attempt, e)
                continue
            issues = check.get("issues", "") if isinstance(check, dict) else ""
            technical_ok = bool(check.get("technical_content_preserved", False)) if isinstance(check, dict) else False
            decorative = bool(check.get("has_decorative_language", True)) if isinstance(check, dict) else True
            relation_count_ok = bool(check.get("relation_count_preserved", False)) if isinstance(check, dict) else False
            causal_ok = bool(check.get("causal_structure_preserved", False)) if isinstance(check, dict) else False
            category_ok = bool(check.get("category_instance_structure_preserved", False)) if isinstance(check, dict) else False
            alias_time_ok = bool(check.get("alias_and_time_markers_preserved", False)) if isinstance(check, dict) else False
            if technical_ok:
                latest_safe = current
            if technical_ok and not decorative and relation_count_ok and causal_ok and category_ok and alias_time_ok:
                return current
        if latest_safe:
            self._log(logging.WARNING, "[3-1] Chunk %s accepted latest technically safe rewrite", idx)
            return latest_safe
        self._log(logging.WARNING, "[3-1] Chunk %s kept input after retries", idx)
        return original

    def _run_stage2(self, idx: int, original: str, retry_limit: int) -> str:
        annotated, items = self._annotate_pronouns(original)
        if not items:
            return original

        guide = self._pronoun_guide(items)
        issues = ""
        previous_replacements = ""
        for attempt in range(1, retry_limit + 1):
            try:
                if attempt == 1:
                    response = self.llm.chat_json(paraphrase_stage2_prompt(original, annotated, guide))
                else:
                    response = self.llm.chat_json(
                        paraphrase_stage2_retry_prompt(
                            original,
                            annotated,
                            guide,
                            previous_replacements,
                            issues,
                        )
                    )
                previous_replacements = self._replacements_to_json(response)
                candidate, replaced_all = self._apply_pronoun_replacements(annotated, items, response)
                self._write_paraphrase_attempt(idx, "stage2", attempt, candidate)
                if not replaced_all:
                    issues = "Not all annotated pronouns were resolved with explicit noun phrases."
                    continue
                check = self.llm.chat_json(paraphrase_stage2_verify_prompt(original, candidate))
            except Exception as e:
                self._log(logging.WARNING, "[3-2] Chunk %s attempt %s LLM error: %s", idx, attempt, e)
                continue
            equivalent = bool(check.get("equivalent", False)) if isinstance(check, dict) else False
            issues = check.get("issues", "") if isinstance(check, dict) else ""
            if equivalent:
                return candidate
        self._log(logging.WARNING, "[3-2] Chunk %s kept input after retries", idx)
        return original

    def _run_stage3(self, idx: int, original: str, retry_limit: int) -> str:
        latest_safe = ""
        current = original
        issues = ""
        for attempt in range(1, retry_limit + 1):
            try:
                if attempt == 1:
                    current = self.llm.chat_text(paraphrase_stage3_prompt(original))
                else:
                    current = self.llm.chat_text(paraphrase_stage3_retry_prompt(original, current, issues))
                self._write_paraphrase_attempt(idx, "stage3", attempt, current)
                check = self.llm.chat_json(paraphrase_stage3_verify_prompt(original, current))
            except Exception as e:
                self._log(logging.WARNING, "[3-3] Chunk %s attempt %s LLM error: %s", idx, attempt, e)
                continue
            issues = check.get("issues", "") if isinstance(check, dict) else ""
            technical_ok = bool(check.get("technical_content_preserved", False)) if isinstance(check, dict) else False
            unresolved = bool(check.get("has_underspecified_references", True)) if isinstance(check, dict) else True
            if technical_ok:
                latest_safe = current
            if technical_ok and not unresolved:
                return current
        if latest_safe:
            self._log(logging.WARNING, "[3-3] Chunk %s accepted latest technically safe rewrite", idx)
            return latest_safe
        self._log(logging.WARNING, "[3-3] Chunk %s kept input after retries", idx)
        return original

    def _run_stage4(self, idx: int, original: str, retry_limit: int) -> str:
        current = original
        issues = ""
        for attempt in range(1, retry_limit + 1):
            try:
                if attempt == 1:
                    current = self.llm.chat_text(paraphrase_stage4_prompt(original))
                else:
                    current = self.llm.chat_text(paraphrase_stage4_retry_prompt(original, current, issues))
                self._write_paraphrase_attempt(idx, "stage4", attempt, current)
                check = self.llm.chat_json(paraphrase_stage4_verify_prompt(original, current))
            except Exception as e:
                self._log(logging.WARNING, "[3-4] Chunk %s attempt %s LLM error: %s", idx, attempt, e)
                continue
            issues = check.get("issues", "") if isinstance(check, dict) else ""
            technical_ok = bool(check.get("technical_content_preserved", False)) if isinstance(check, dict) else False
            clear_spo = bool(check.get("clear_spo", False)) if isinstance(check, dict) else False
            one_relation = bool(check.get("one_relation_per_sentence", False)) if isinstance(check, dict) else False
            roles_ok = bool(check.get("argument_roles_preserved", False)) if isinstance(check, dict) else False
            semantics_ok = bool(check.get("predicate_semantics_preserved", False)) if isinstance(check, dict) else False
            if technical_ok and clear_spo and one_relation and roles_ok and semantics_ok:
                return current
        self._log(logging.WARNING, "[3-4] Chunk %s falling back to stage2 output", idx)
        return original

    def _paraphrase_single_chunk(self, idx: int, chunk_path: Path, retry_limit: int) -> Path:
        original = self._read_text(chunk_path)
        out = self.paraphrase_dir / f"chunk_{idx:04d}.txt"
        if not original.strip():
            out.write_text("", encoding="utf-8")
            return out

        current = original
        current = self._run_stage1(idx, current, retry_limit)
        current = self._run_stage2(idx, current, retry_limit)
        # current = self._run_stage3(idx, current, retry_limit)
        current = self._run_stage4(idx, current, retry_limit)
        out.write_text(current, encoding="utf-8")
        return out

    def node_paraphrasing(self) -> None:
        self._log(logging.INFO, "[3] Paraphrasing start")
        retry_limit = 3
        batch = int(self.config.get("parallel_batch_size", 6))
        self.paraphrase_files = []

        results: Dict[int, Path] = {}
        with ThreadPoolExecutor(max_workers=batch) as executor:
            futures = {
                executor.submit(self._paraphrase_single_chunk, idx, path, retry_limit): idx
                for idx, path in enumerate(self.chunk_files)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    self._log(logging.ERROR, "[3] Chunk %s unhandled error: %s", idx, e)

        self.paraphrase_files = [results[i] for i in sorted(results)]
        if len(self.paraphrase_files) < len(self.chunk_files):
            self._log(
                logging.WARNING,
                "[3] Only %s/%s chunks paraphrased successfully",
                len(self.paraphrase_files),
                len(self.chunk_files),
            )
        self._log(logging.INFO, "[3] Paraphrasing completed: %s files", len(self.paraphrase_files))

    # ------------------------------------------------------------------
    # Node 4: Entity Extraction
    # ------------------------------------------------------------------

    def _normalize_entities(self, chunk_name: str, payload: Any) -> List[Dict[str, str]]:
        if not isinstance(payload, dict) or not isinstance(payload.get("entities"), list):
            return []
        normalized: List[Dict[str, str]] = []
        seen: set[Tuple[str, str]] = set()
        for item in payload["entities"]:
            if not isinstance(item, dict):
                continue
            name = clean_text(item.get("name", ""))
            description = clean_text(item.get("description", ""))
            if not name or not description:
                continue
            key = (chunk_name, name.lower())
            if key in seen:
                continue
            seen.add(key)
            normalized.append({
                "source_chunk": chunk_name,
                "name": name,
                "description": description,
            })
        return normalized

    def _extract_chunk_entities(self, path: Path, retries: int) -> List[Dict[str, str]]:
        chunk_text = self._read_text(path)
        chunk_name = path.stem
        for _ in range(retries):
            try:
                parsed = self.llm.chat_json(entity_extraction_prompt(chunk_name, chunk_text))
            except Exception as e:
                self._log(logging.WARNING, "[4] Entity extraction failed for %s: %s", chunk_name, e)
                continue
            entities = self._normalize_entities(chunk_name, parsed)
            if entities or (isinstance(parsed, dict) and parsed.get("entities") == []):
                return entities
        self._log(
            logging.WARNING,
            "[4] Entity extraction exhausted %s retries for %s, returning empty",
            retries,
            chunk_name,
        )
        return []

    def _extract_chunk_entities_from_triplets(self, path: Path, retries: int) -> List[Dict[str, str]]:
        chunk_text = self._read_text(path)
        chunk_name = path.stem
        chunk_triplets = [
            {
                "subject": str(item.get("subject", "")),
                "predicate": str(item.get("predicate", "")),
                "object": str(item.get("object", "")),
            }
            for item in self.triplets
            if item.get("source_chunk") == chunk_name
        ]
        if not chunk_triplets:
            return self._extract_chunk_entities(path, retries)
        for _ in range(retries):
            try:
                parsed = self.llm.chat_json(
                    entity_inventory_from_triplets_prompt(chunk_name, chunk_text, chunk_triplets)
                )
            except Exception as e:
                self._log(logging.WARNING, "[4] Entity backfill failed for %s: %s", chunk_name, e)
                continue
            entities = self._normalize_entities(chunk_name, parsed)
            if entities or (isinstance(parsed, dict) and parsed.get("entities") == []):
                return entities
        return self._extract_chunk_entities(path, retries)

    def _refresh_triplet_descriptions(self) -> None:
        for row in self.triplets:
            row["subject_description"] = self._entity_description_context(
                row.get("source_chunk", ""),
                row.get("subject", ""),
            )
            row["object_description"] = self._entity_description_context(
                row.get("source_chunk", ""),
                row.get("object", ""),
            )

    def node_entity_extraction(self) -> None:
        self._log(logging.INFO, "[4] Entity Extraction start")
        retries = int(self.config["retry"].get("entity_extraction", 3))
        batch = int(self.config.get("parallel_batch_size", 6))
        ordered_results: List[List[Dict[str, str]]] = [[] for _ in self.paraphrase_files]

        with ThreadPoolExecutor(max_workers=batch) as executor:
            futures = {
                executor.submit(self._extract_chunk_entities_from_triplets, path, retries): idx
                for idx, path in enumerate(self.paraphrase_files)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    ordered_results[idx] = future.result()
                except Exception as e:
                    self._log(logging.ERROR, "[4] Chunk %s entity extraction error: %s", idx, e)

        self.entities = []
        self.entities_by_chunk = {}
        for entities in ordered_results:
            for entity in entities:
                self.entities.append(entity)
                self.entities_by_chunk.setdefault(entity["source_chunk"], []).append(entity)

        self._refresh_triplet_descriptions()
        self._build_entity_memory()
        self._write_json(self.intermediate / "entities.json", {"entities": self.entities})
        self._write_json(self.intermediate / "entityMemory.json", self.entity_memory)
        self._write_json(self.intermediate / "triplets.json", self.triplets)
        self._log(logging.INFO, "[4] Entity Extraction completed: %s entities", len(self.entities))

    # ------------------------------------------------------------------
    # Node 5: Triplet Extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_entity_key(name: str) -> str:
        return re.sub(r"\s+", " ", clean_text(name).casefold())

    def _sentence_entities(self, chunk_name: str, sentence: str) -> List[Dict[str, str]]:
        _ws = r'\s+'
        sentence_text = " " + re.sub(_ws, " ", sentence.casefold()) + " "
        matched: List[Dict[str, str]] = []
        for entity in self.entities_by_chunk.get(chunk_name, []):
            name = entity["name"]
            normalized_name = self._normalize_entity_key(name)
            if not normalized_name:
                continue
            _escaped = re.escape(normalized_name)
            pattern = r"(?<!\w)" + _escaped + r"(?!\w)"
            if re.search(pattern, sentence_text):
                matched.append(entity)
        return matched

    def _entity_description_context(self, chunk_name: str, entity_name: str) -> str:
        target = self._normalize_entity_key(entity_name)
        if not target:
            return ""
        for item in self.entities_by_chunk.get(chunk_name, []):
            if self._normalize_entity_key(item["name"]) == target:
                return item["description"]
        for item in self.entities:
            if self._normalize_entity_key(item["name"]) == target:
                return item["description"]
        return ""

    @staticmethod
    def _evidence_weight(kind: str) -> int:
        return EVIDENCE_KIND_WEIGHTS.get(kind, 0)

    def _classify_evidence_kind(self, entity: str, text: str) -> str:
        name = clean_text(entity)
        context = clean_text(text)
        if not name or not context:
            return "relation_context"

        escaped = re.escape(name)
        explicit_patterns = [
            rf"\b{escaped}\b\s+(?:is|was|are|were)\s+(?:an?|the)\b",
            rf"\b{escaped}\b\s+(?:refers to|denotes|represents)\b",
        ]
        for pattern in explicit_patterns:
            if re.search(pattern, context, flags=re.IGNORECASE):
                return "explicit_definition"

        apposition_patterns = [
            rf"\b{escaped}\b\s*,\s*(?:an?|the)\b",
            rf"(?:an?|the)\s+[^.(),;]{{1,80}},\s*\b{escaped}\b",
        ]
        for pattern in apposition_patterns:
            if re.search(pattern, context, flags=re.IGNORECASE):
                return "apposition"

        lowered = name.casefold()
        if (
            re.search(r"\bCVE-\d{4}-\d{4,7}\b", name)
            or re.search(r"\b(?:iOS|Android|Windows|macOS|Linux)\b", name, flags=re.IGNORECASE)
            or any(token in lowered for token in ("spyware", "ransomware", "malware", "phishing", "exploit"))
        ):
            return "name_pattern"

        if len(context.split()) <= 28:
            return "entity_description"
        return "relation_context"

    def _remember_entity_evidence(
        self,
        entity_name: str,
        chunk_name: str,
        text: str,
        evidence_kind: str,
    ) -> None:
        key = self._normalize_entity_key(entity_name)
        if not key:
            return
        evidence_text = clean_text(text)
        if not evidence_text:
            return

        memory = self.entity_memory.setdefault(
            key,
            {
                "name": clean_text(entity_name),
                "aliases": set(),
                "evidence": [],
                "best_evidence": {"kind": "", "weight": 0, "text": "", "source_chunk": ""},
                "resolved_types": {},
            },
        )
        memory["aliases"].add(clean_text(entity_name))
        weight = self._evidence_weight(evidence_kind)
        entry = {
            "kind": evidence_kind,
            "weight": weight,
            "text": evidence_text,
            "source_chunk": chunk_name,
        }
        if entry not in memory["evidence"]:
            memory["evidence"].append(entry)
        best = memory["best_evidence"]
        if weight > int(best.get("weight", 0)) or (
            weight == int(best.get("weight", 0)) and len(evidence_text) > len(str(best.get("text", "")))
        ):
            memory["best_evidence"] = entry

    def _remember_resolved_entity_type(
        self,
        entity_name: str,
        class_uri: str,
        class_name: str,
    ) -> None:
        key = self._normalize_entity_key(entity_name)
        if not key or not class_uri:
            return
        memory = self.entity_memory.setdefault(
            key,
            {
                "name": clean_text(entity_name),
                "aliases": set(),
                "evidence": [],
                "best_evidence": {"kind": "", "weight": 0, "text": "", "source_chunk": ""},
                "resolved_types": {},
            },
        )
        resolved = memory.setdefault("resolved_types", {})
        stats = resolved.setdefault(class_uri, {"class_name": class_name, "count": 0})
        stats["count"] += 1
        if class_name:
            stats["class_name"] = class_name

    def _build_entity_memory(self) -> None:
        self.entity_memory = {}

        for chunk_name, chunk_entities in self.entities_by_chunk.items():
            for entity in chunk_entities:
                description = str(entity.get("description", "") or "")
                evidence_kind = self._classify_evidence_kind(entity.get("name", ""), description)
                self._remember_entity_evidence(
                    str(entity.get("name", "") or ""),
                    chunk_name,
                    description,
                    evidence_kind,
                )

        for row in self.triplets:
            sentence = str(row.get("source_sentence", "") or "")
            chunk_name = str(row.get("source_chunk", "") or "")
            if row.get("subject"):
                self._remember_entity_evidence(
                    str(row["subject"]),
                    chunk_name,
                    sentence,
                    "relation_context",
                )
            if row.get("object"):
                self._remember_entity_evidence(
                    str(row["object"]),
                    chunk_name,
                    sentence,
                    "relation_context",
                )

        for memory in self.entity_memory.values():
            memory["aliases"] = sorted(memory["aliases"])

    def _entity_memory_summary(self, entity_name: str) -> str:
        key = self._normalize_entity_key(entity_name)
        if not key:
            return ""
        memory = self.entity_memory.get(key)
        if not memory:
            return ""

        lines: List[str] = []
        aliases = [alias for alias in memory.get("aliases", []) if alias and alias != entity_name]
        if aliases:
            lines.append("Known aliases: " + ", ".join(aliases[:3]))

        best = memory.get("best_evidence") or {}
        if best.get("text"):
            lines.append(
                "Strongest typing evidence "
                f"({best.get('kind', 'unknown')}): {best.get('text', '')}"
            )

        resolved_types = memory.get("resolved_types") or {}
        if resolved_types:
            ranked = sorted(
                resolved_types.items(),
                key=lambda item: int(item[1].get("count", 0)),
                reverse=True,
            )
            consensus = ranked[0][1]
            lines.append(
                "Current document-level type consensus: "
                f"{consensus.get('class_name') or ranked[0][0]} "
                f"(seen {consensus.get('count', 0)} time(s))"
            )

        evidence = memory.get("evidence") or []
        if evidence:
            ranked_evidence = sorted(
                evidence,
                key=lambda item: (int(item.get("weight", 0)), len(str(item.get("text", "")))),
                reverse=True,
            )
            for item in ranked_evidence[:2]:
                text = str(item.get("text", ""))
                if best.get("text") and text == best.get("text"):
                    continue
                lines.append(f"Additional evidence ({item.get('kind', 'unknown')}): {text}")
                if len(lines) >= 4:
                    break

        return "\n".join(lines)

    def _extract_chunk_triplets(self, path: Path, retries: int) -> List[Dict[str, Any]]:
        """Extract triplets from one paraphrased chunk. Thread-safe (only uses self.llm)."""
        chunk_text = self._read_text(path)
        chunk_name = path.stem
        chunk_triplets: List[Dict[str, Any]] = []
        chunk_entities = self.entities_by_chunk.get(chunk_name, [])
        for sentence in self._split_sentences(chunk_text):
            sentence_entities = self._sentence_entities(chunk_name, sentence)
            extracted = None
            for _ in range(retries):
                try:
                    parsed = self.llm.chat_json(
                        triplet_prompt(chunk_text, sentence, chunk_entities, sentence_entities)
                    )
                except Exception as e:
                    self._log(logging.WARNING, "[5] LLM error for sentence: %s", e)
                    continue
                if isinstance(parsed, dict) and isinstance(parsed.get("triplets"), list):
                    extracted = parsed["triplets"]
                    break
            if not extracted:
                self._log(logging.WARNING, "[5] Skip sentence (invalid triplet JSON): %s", sentence[:80])
                continue
            for item in extracted:
                if not isinstance(item, dict):
                    continue
                subject = clean_text(item.get("subject", ""))
                predicate = clean_text(item.get("predicate", ""))
                obj = clean_text(item.get("object", ""))
                if not subject or not predicate or not obj:
                    continue
                # id assigned later in order
                chunk_triplets.append({
                    "source_sentence": sentence,
                    "source_chunk": chunk_name,
                    "subject": subject,
                    "subject_description": self._entity_description_context(chunk_name, subject),
                    "predicate": predicate,
                    "object": obj,
                    "object_description": self._entity_description_context(chunk_name, obj),
                    "isSubjectIoC": False,
                    "isObjectIoC": False,
                    "subjectIoCType": "",
                    "objectIoCType": "",
                })
        return chunk_triplets

    @staticmethod
    def _word_count(text: str) -> int:
        return len([part for part in re.split(r"\s+", clean_text(text)) if part])

    @staticmethod
    def _contains_relation_signal(text: str) -> bool:
        lowered = f" {clean_text(text).casefold()} "
        signals = (
            " in ",
            " for ",
            " by ",
            " against ",
            " from ",
            " with ",
            " such as ",
            " including ",
            " used to ",
            " designed to ",
        )
        return any(signal in lowered for signal in signals)

    @staticmethod
    def _looks_like_product_version(text: str) -> bool:
        cleaned = clean_text(text)
        if not cleaned:
            return False
        if re.search(r"\b(?:version|ver\.)\s*\d", cleaned, flags=re.IGNORECASE):
            return True
        return bool(
            re.search(
                r"\b(?:iOS|Android|Windows|macOS|Linux|Chrome|Safari|Firefox|Exchange|Office)\s+\d+(?:\.\d+){0,3}\b",
                cleaned,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _looks_like_named_org_or_team(text: str) -> bool:
        cleaned = clean_text(text)
        if not cleaned:
            return False
        if "(" in cleaned and ")" in cleaned:
            return True
        tokens = cleaned.split()
        if len(tokens) >= 2 and all(token[:1].isupper() or token.lower() in {"the", "of", "and", "&"} for token in tokens):
            return True
        return any(
            keyword in cleaned.casefold()
            for keyword in (" lab", " team", " group", " inc", " ltd", " llc", " corp", " corporation", " company")
        )

    @staticmethod
    def _looks_like_fixed_malware_or_tool_name(text: str) -> bool:
        cleaned = clean_text(text)
        lowered = cleaned.casefold()
        return any(
            lowered.endswith(suffix)
            for suffix in (
                " spyware",
                " ransomware",
                " malware",
                " trojan",
                " backdoor",
                " loader",
                " stealer",
                " botnet",
                " exploit kit",
            )
        )

    def _is_protected_entity_span(self, text: str) -> bool:
        cleaned = clean_text(text)
        if not cleaned:
            return True
        if re.search(r"\bCVE-\d{4}-\d{4,7}\b", cleaned):
            return True
        if self._looks_like_product_version(cleaned):
            return True
        if self._looks_like_named_org_or_team(cleaned):
            return True
        if self._looks_like_fixed_malware_or_tool_name(cleaned):
            return True
        if self._ioc_matches(cleaned):
            return True
        return False

    def _should_consider_span_normalization(self, text: str) -> bool:
        cleaned = clean_text(text)
        if not cleaned:
            return False
        if self._is_protected_entity_span(cleaned):
            return False
        max_words_without_signal = int(
            self.config.get("span_normalization", {}).get("max_words_without_signal", 3)
        )
        if self._contains_relation_signal(cleaned):
            return True
        return self._word_count(cleaned) > max_words_without_signal

    def _normalize_triplet_span(
        self,
        row: Dict[str, Any],
        role: str,
    ) -> Tuple[str, List[Dict[str, str]], Optional[Dict[str, Any]]]:
        span = str(row.get(role, "") or "")
        if not self._should_consider_span_normalization(span):
            return span, [], None

        try:
            parsed = self.llm.chat_json(
                span_normalization_prompt(
                    sentence=str(row.get("source_sentence", "") or ""),
                    subject=str(row.get("subject", "") or ""),
                    predicate=str(row.get("predicate", "") or ""),
                    obj=str(row.get("object", "") or ""),
                    role=role,
                    span=span,
                ),
                trace_context={
                    "event": "span_normalization",
                    "role": role,
                    "span": span,
                    "row_id": row.get("id"),
                },
            )
        except Exception as e:
            self._log(logging.WARNING, "[5.5] Span normalization failed for %s '%s': %s", role, span, e)
            return span, [], None

        if not isinstance(parsed, dict):
            return span, [], None

        decision = str(parsed.get("decision", "keep") or "keep").strip().lower()
        canonical = clean_text(parsed.get("canonical_span", "") or span)
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        threshold = float(self.config.get("span_normalization", {}).get("min_confidence", 0.8))
        if confidence < threshold or decision not in {"keep", "rewrite_head", "split"}:
            return span, [], None
        if decision == "keep" or not canonical:
            return span, [], None

        qualifiers: List[Dict[str, str]] = []
        raw_qualifiers = parsed.get("qualifiers", [])
        if isinstance(raw_qualifiers, list):
            for item in raw_qualifiers[:3]:
                if not isinstance(item, dict):
                    continue
                relation_hint = clean_text(item.get("relation_hint", ""))
                value = clean_text(item.get("value", ""))
                if relation_hint and value and value != canonical:
                    qualifiers.append({"relation_hint": relation_hint, "value": value})

        metadata = {
            "decision": decision,
            "original_span": span,
            "canonical_span": canonical,
            "qualifiers": qualifiers,
            "confidence": confidence,
            "reason": str(parsed.get("reason", "") or ""),
        }
        return canonical, qualifiers if decision == "split" else [], metadata

    def _recompose_triplet_from_qualifiers(
        self,
        row: Dict[str, Any],
        canonical_span: str,
        qualifiers: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        extra_rows: List[Dict[str, Any]] = []
        for item in qualifiers:
            relation_hint = clean_text(item.get("relation_hint", ""))
            value = clean_text(item.get("value", ""))
            if not relation_hint or not value:
                continue
            judge = self._judge_qualifier_node_worthiness(
                row=row,
                canonical_span=canonical_span,
                relation_hint=relation_hint,
                qualifier_value=value,
            )
            self._log_qualifier_judge(
                row_id=row.get("id"),
                canonical_span=canonical_span,
                relation_hint=relation_hint,
                qualifier_value=value,
                judge=judge,
            )
            if not self._should_promote_qualifier(judge):
                continue
            extra_rows.append({
                "id": -1,
                "source_sentence": row["source_sentence"],
                "source_chunk": row["source_chunk"],
                "subject": canonical_span,
                "subject_description": "",
                "predicate": relation_hint,
                "object": value,
                "object_description": "",
                "isSubjectIoC": False,
                "isObjectIoC": False,
                "subjectIoCType": "",
                "objectIoCType": "",
            })
        return extra_rows

    def _judge_qualifier_node_worthiness(
        self,
        row: Dict[str, Any],
        canonical_span: str,
        relation_hint: str,
        qualifier_value: str,
    ) -> Dict[str, Any]:
        if not bool(self.config.get("qualifier_judge", {}).get("enabled", True)):
            return {
                "is_node_worthy": True,
                "category": "entity",
                "suggested_handling": "promote_node",
                "confidence": 1.0,
                "reason": "Qualifier judge disabled",
            }

        qualifier_memory = self._entity_memory_summary(qualifier_value)
        try:
            parsed = self.llm.chat_json(
                qualifier_node_worthiness_prompt(
                    sentence=str(row.get("source_sentence", "") or ""),
                    subject=str(row.get("subject", "") or ""),
                    predicate=str(row.get("predicate", "") or ""),
                    obj=str(row.get("object", "") or ""),
                    canonical_span=canonical_span,
                    role="subject" if str(row.get("subject", "")) == canonical_span else "object",
                    relation_hint=relation_hint,
                    qualifier_value=qualifier_value,
                    entity_memory=qualifier_memory,
                ),
                trace_context={
                    "event": "qualifier_node_worthiness",
                    "row_id": row.get("id"),
                    "canonical_span": canonical_span,
                    "relation_hint": relation_hint,
                    "qualifier_value": qualifier_value,
                },
            )
        except Exception as e:
            self._log(
                logging.WARNING,
                "[5.6] Qualifier judge failed for '%s' (%s): %s",
                qualifier_value,
                relation_hint,
                e,
            )
            return {
                "is_node_worthy": False,
                "category": "noise",
                "suggested_handling": "keep_as_qualifier",
                "confidence": 0.0,
                "reason": f"Judge failure: {e}",
            }

        if not isinstance(parsed, dict):
            return {
                "is_node_worthy": False,
                "category": "noise",
                "suggested_handling": "keep_as_qualifier",
                "confidence": 0.0,
                "reason": "Invalid qualifier judge response",
            }
        return parsed

    def _should_promote_qualifier(self, judge: Dict[str, Any]) -> bool:
        confidence = float(judge.get("confidence", 0.0) or 0.0)
        threshold = float(self.config.get("qualifier_judge", {}).get("min_confidence", 0.75))
        handling = str(judge.get("suggested_handling", "") or "").strip().lower()
        is_node_worthy = bool(judge.get("is_node_worthy", False))
        category = str(judge.get("category", "") or "").strip().lower()
        return (
            is_node_worthy
            and handling == "promote_node"
            and category == "entity"
            and confidence >= threshold
        )

    def _log_qualifier_judge(
        self,
        row_id: Any,
        canonical_span: str,
        relation_hint: str,
        qualifier_value: str,
        judge: Dict[str, Any],
    ) -> None:
        self._log(
            logging.INFO,
            (
                '[5.6][judge] triplet=%s | canonical="%s" | relation_hint=%s | qualifier="%s" '
                '| node_worthy=%s | category=%s | handling=%s | confidence=%.2f | reason=%s'
            ),
            row_id,
            canonical_span,
            relation_hint,
            qualifier_value,
            bool(judge.get("is_node_worthy", False)),
            str(judge.get("category", "") or ""),
            str(judge.get("suggested_handling", "") or ""),
            float(judge.get("confidence", 0.0) or 0.0),
            str(judge.get("reason", "") or ""),
        )

    def _log_span_normalization(
        self,
        row_id: Any,
        role: str,
        metadata: Dict[str, Any],
        derived_count: int,
    ) -> None:
        decision = str(metadata.get("decision", "") or "")
        original_span = str(metadata.get("original_span", "") or "")
        canonical_span = str(metadata.get("canonical_span", "") or "")
        confidence = float(metadata.get("confidence", 0.0) or 0.0)
        qualifiers = metadata.get("qualifiers", []) or []
        reason = str(metadata.get("reason", "") or "")

        qualifier_text = ", ".join(
            f'{item.get("relation_hint", "")}:{item.get("value", "")}'
            for item in qualifiers
            if isinstance(item, dict)
        ) or "-"

        self._log(
            logging.INFO,
            '[5.5][%s] triplet=%s role=%s | "%s" -> "%s" | confidence=%.2f | qualifiers=%s | derived=%s | reason=%s',
            decision,
            row_id,
            role,
            original_span,
            canonical_span,
            confidence,
            qualifier_text,
            derived_count,
            reason,
        )

    def node_triplet_normalization(self) -> None:
        self._log(logging.INFO, "[5.5] Triplet Span Normalization start")
        if not bool(self.config.get("span_normalization", {}).get("enabled", True)):
            self._log(logging.INFO, "[5.5] Triplet Span Normalization disabled")
            return

        normalized_rows: List[Dict[str, Any]] = []
        next_id = 0
        for original in self.triplets:
            row = dict(original)
            extra_rows: List[Dict[str, Any]] = []
            normalizations: List[Dict[str, Any]] = []

            for role in ("subject", "object"):
                canonical, qualifiers, metadata = self._normalize_triplet_span(row, role)
                if metadata:
                    row[role] = canonical
                    normalizations.append({"role": role, **metadata})
                    self._log_span_normalization(
                        row.get("id"),
                        role,
                        metadata,
                        len(qualifiers),
                    )
                    if qualifiers:
                        extra_rows.extend(
                            self._recompose_triplet_from_qualifiers(row, canonical, qualifiers)
                        )

            row["id"] = next_id
            next_id += 1
            if normalizations:
                row["span_normalization"] = normalizations
            normalized_rows.append(row)

            for extra in extra_rows:
                extra["id"] = next_id
                next_id += 1
                extra["span_normalization"] = [{
                    "role": "derived",
                    "decision": "recomposed",
                    "original_span": "",
                    "canonical_span": extra["subject"],
                    "qualifiers": [],
                    "confidence": 1.0,
                    "reason": "Derived from high-confidence split qualifier",
                }]
                normalized_rows.append(extra)

        self.triplets = normalized_rows
        self._write_json(self.intermediate / "triplets.json", self.triplets)
        self._log(logging.INFO, "[5.5] Triplet Span Normalization completed: %s triplets", len(self.triplets))

    def node_triplet_extraction(self) -> None:
        self._log(logging.INFO, "[5] Triplet Extraction start")
        retries = int(self.config["retry"]["triplet_extraction"])
        batch = int(self.config.get("parallel_batch_size", 6))

        # Process chunks in parallel; preserve chunk order for deterministic triplet IDs
        ordered_results: List[List[Dict[str, Any]]] = [[] for _ in self.paraphrase_files]
        with ThreadPoolExecutor(max_workers=batch) as executor:
            futures = {
                executor.submit(self._extract_chunk_triplets, path, retries): idx
                for idx, path in enumerate(self.paraphrase_files)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    ordered_results[idx] = future.result()
                except Exception as e:
                    self._log(logging.ERROR, "[5] Chunk %s extraction error: %s", idx, e)

        # Assign sequential IDs in chunk order
        triplets: List[Dict[str, Any]] = []
        next_id = 0
        for chunk_triplets in ordered_results:
            for trip in chunk_triplets:
                trip["id"] = next_id
                next_id += 1
                triplets.append(trip)

        self.triplets = triplets
        self._write_json(self.intermediate / "triplets.json", self.triplets)
        self._log(logging.INFO, "[5] Triplet Extraction completed: %s triplets", len(self.triplets))

    # ------------------------------------------------------------------
    # Node 6: IoC Detection & Rearm
    # ------------------------------------------------------------------

    def node_ioc_detection(self) -> None:
        self._log(logging.INFO, "[6] IoC Detection start")
        for row in self.triplets:
            for field in ("subject", "object"):
                value = row[field]
                if not value:
                    continue
                matches = self._ioc_matches(value)
                if not matches:
                    continue
                for match in matches:
                    candidate = str(match.get("value", "") or "")
                    detected_type = str(match.get("type", "") or "")
                    if not candidate:
                        continue
                    parsed = self.llm.chat_json(ioc_prompt(candidate, value, detected_type))
                    if not isinstance(parsed, dict):
                        continue
                    if bool(parsed.get("is_ioc", False)):
                        rearmed = parsed.get("rearmed_value") or value
                        ioc_type = str(parsed.get("ioc_type", "") or detected_type)
                        if field == "subject":
                            row["isSubjectIoC"] = True
                            row["subject"] = rearmed
                            row["subjectIoCType"] = ioc_type
                        else:
                            row["isObjectIoC"] = True
                            row["object"] = rearmed
                            row["objectIoCType"] = ioc_type
                        self._log(logging.INFO, "[6] IoC detected (%s/%s): %s -> %s", field, ioc_type or "unknown", value, rearmed)
                        break  # first confirmed IoC per field is sufficient

        # Write once after processing all triplets
        self._write_json(self.intermediate / "triplets.json", self.triplets)
        self._log(logging.INFO, "[6] IoC Detection completed")

    @staticmethod
    def _regex_ioc_matches(value: str) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        patterns = [
            (r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "ip4"),
            (r"\b(?:https?|hxxps?)://[^\s)\]\"']+", "url"),
            (r"\b[A-Fa-f0-9]{32}\b", "md5"),
            (r"\b[A-Fa-f0-9]{40}\b", "sha1"),
            (r"\b[A-Fa-f0-9]{64}\b", "sha256"),
            (r"\bCVE-\d{4}-\d{4,7}\b", "cve"),
            (r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", "email"),
            (r"\bT\d{4}(?:\.\d{3})?\b", "ttp"),
        ]
        seen: set[Tuple[str, str]] = set()
        for pat, ioc_type in patterns:
            for m in re.findall(pat, value):
                hit = m if isinstance(m, str) else m[0]
                key = (ioc_type, hit)
                if hit and key not in seen:
                    seen.add(key)
                    results.append({"type": ioc_type, "value": hit, "raw": hit})
        return results

    def _ioc_matches(self, value: str) -> List[Dict[str, str]]:
        results = self._regex_ioc_matches(value)
        try:
            if not self._ioc_searcher_loaded:
                try:
                    from iocsearcher.searcher import Searcher  # type: ignore
                    self._ioc_searcher = Searcher()
                except ImportError:
                    self._ioc_searcher = None
                except Exception as e:
                    self._log(logging.DEBUG, "[6] IOC searcher initialization failed: %s", e)
                    self._ioc_searcher = None
                self._ioc_searcher_loaded = True
            if self._ioc_searcher is None:
                return results
            raw_hits = self._ioc_searcher.search_raw(value) or []
            seen = {(item["type"], item["value"]) for item in results}
            for hit in raw_hits:
                if not isinstance(hit, (list, tuple)) or len(hit) < 2:
                    continue
                ioc_type = str(hit[0] or "").strip()
                normalized = str(hit[1] or "").strip()
                raw_value = str(hit[3] or normalized).strip() if len(hit) > 3 else normalized
                if not (ioc_type and normalized):
                    continue
                key = (ioc_type, normalized)
                if key in seen:
                    continue
                seen.add(key)
                results.append({"type": ioc_type, "value": normalized, "raw": raw_value})
        except Exception as e:
            self._log(logging.WARNING, "[6] IOC searcher search failed: %s", e)
        return results

    # ------------------------------------------------------------------
    # Node 7: Triplet Type Matching
    # ------------------------------------------------------------------

    def node_type_matching(self) -> None:
        self._log(logging.INFO, "[7] Triplet Type Matching start")
        self._reset_type_matching_progress()
        mcp = self._try_create_mcp()
        if mcp is None:
            self._log(logging.WARNING, "[7] MCP unavailable; typed triplets will have empty class info")
            self.all_typed_triplets = [self._empty_typed_row(row) for row in self.triplets]
            self.typed_triplets = []
            self._write_json(self.intermediate / "typedTriplets.json", self.all_typed_triplets)
            return

        with mcp:
            self._verify_mcp_tools(mcp)
            self._select_root_node(mcp)
            self._log(logging.INFO, "[7-0] Root node: %s (%s)",
                      self.report_root["name"], self.report_root["class_uri"])
            self._warm_ioc_type_mappings(mcp)

            typed_rows: List[Optional[Dict[str, Any]]] = []
            for i, row in enumerate(self.triplets):
                try:
                    typed_rows.append(self._type_match_triplet(mcp, row))
                except Exception as e:
                    self._log(logging.ERROR, "[7] Triplet %s type-match error: %s", i, e)
                    fallback = self._empty_typed_row(self.triplets[i])
                    typed_rows.append(fallback)
                    progress = self._advance_type_matching_progress()
                    self._log_type_matching_result(self.triplets[i], fallback, progress)
            typed_candidates = [r for r in typed_rows if r is not None]

        kept_rows: List[Dict[str, Any]] = []
        dropped_rows: List[Dict[str, Any]] = []
        for row in typed_candidates:
            if self._is_ontology_compliant_triplet(row):
                kept_rows.append(row)
            else:
                dropped_rows.append(row)

        self.all_typed_triplets = typed_candidates
        self.typed_triplets = kept_rows
        self._write_json(self.intermediate / "typedTriplets.json", self.all_typed_triplets)
        self._write_json(self.intermediate / "entityMemory.json", self.entity_memory)
        self._log(
            logging.INFO,
            "[7] Type Matching completed. kept=%s dropped=%s MCP calls: type=%s, property=%s",
            len(self.typed_triplets),
            len(dropped_rows),
            self.mcp_call_count["type_matching"],
            self.mcp_call_count["property_matching"],
        )

    def _empty_typed_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "source_sentence": row["source_sentence"],
            "source_chunk": row["source_chunk"],
            "subject": row["subject"],
            "subject_description": row.get("subject_description", ""),
            "subject_class_uri": "",
            "subject_class_name": "",
            "predicate": row["predicate"],
            "predicate_uri": "",
            "predicate_is_inverse": False,
            "object": row["object"],
            "object_description": row.get("object_description", ""),
            "object_is_literal": False,
            "object_class_uri": "",
            "object_class_name": "",
            "isSubjectIoC": row["isSubjectIoC"],
            "isObjectIoC": row["isObjectIoC"],
            "subjectIoCType": row.get("subjectIoCType", ""),
            "objectIoCType": row.get("objectIoCType", ""),
        }

    @staticmethod
    def _ioc_type_context(ioc_type: str) -> str:
        descriptions = {
            "url": "URL indicator of compromise",
            "fqdn": "fully qualified domain name indicator of compromise",
            "ip4": "IPv4 address indicator of compromise",
            "ip6": "IPv6 address indicator of compromise",
            "ip4Net": "IPv4 subnet or CIDR network indicator of compromise",
            "md5": "MD5 file hash indicator of compromise",
            "sha1": "SHA1 file hash indicator of compromise",
            "sha256": "SHA256 file hash indicator of compromise",
            "email": "email address indicator of compromise",
            "bitcoin": "Bitcoin blockchain address indicator of compromise",
            "bitcoincash": "Bitcoin Cash blockchain address indicator of compromise",
            "cardano": "Cardano blockchain address indicator of compromise",
            "dashcoin": "Dash blockchain address indicator of compromise",
            "dogecoin": "Dogecoin blockchain address indicator of compromise",
            "ethereum": "Ethereum blockchain address indicator of compromise",
            "litecoin": "Litecoin blockchain address indicator of compromise",
            "monero": "Monero blockchain address indicator of compromise",
            "ripple": "Ripple blockchain address indicator of compromise",
            "solana": "Solana blockchain address indicator of compromise",
            "stellar": "Stellar blockchain address indicator of compromise",
            "tezos": "Tezos blockchain address indicator of compromise",
            "tron": "Tron blockchain address indicator of compromise",
            "zcash": "Zcash blockchain address indicator of compromise",
            "phoneNumber": "phone number indicator of compromise",
            "copyright": "copyright string extracted as an indicator",
            "cve": "CVE vulnerability identifier indicator of compromise",
            "onionAddress": "Tor v3 onion service address indicator of compromise",
            "facebookHandle": "Facebook social handle indicator",
            "githubHandle": "GitHub social handle indicator",
            "instagramHandle": "Instagram social handle indicator",
            "linkedinHandle": "LinkedIn social handle indicator",
            "pinterestHandle": "Pinterest social handle indicator",
            "telegramHandle": "Telegram social handle indicator",
            "twitterHandle": "Twitter social handle indicator",
            "whatsappHandle": "WhatsApp social handle indicator",
            "youtubeHandle": "YouTube handle indicator",
            "youtubeChannel": "YouTube channel identifier indicator",
            "googleAdsense": "Google AdSense identifier indicator",
            "googleAnalytics": "Google Analytics identifier indicator",
            "googleTagManager": "Google Tag Manager identifier indicator",
            "webmoney": "WebMoney payment address indicator",
            "icp": "Chinese Internet Content Provider license identifier",
            "iban": "IBAN bank account number identifier",
            "trademark": "trademark string extracted as an indicator",
            "uuid": "UUID identifier",
            "packageName": "Android package name identifier",
            "ttp": "MITRE ATT&CK technique identifier",
            "nif": "Spanish NIF identifier",
            "tox": "TOX identifier",
            "arn": "Amazon Resource Name identifier",
        }
        return descriptions.get(ioc_type, f"{ioc_type} indicator of compromise")

    def _match_ioc_type_class(
        self, mcp: MCPStdioClient, ioc_type: str
    ) -> Tuple[str, str]:
        cache_key = ioc_type.strip()
        with self._cache_lock:
            if cache_key in self._ioc_type_class_cache:
                return self._ioc_type_class_cache[cache_key]

        result = self._match_entity_class(
            mcp,
            cache_key,
            f"IoC type keyword: {cache_key}. Meaning: {self._ioc_type_context(cache_key)}.",
        )
        with self._cache_lock:
            self._ioc_type_class_cache[cache_key] = result
        return result

    def _warm_ioc_type_mappings(self, mcp: MCPStdioClient) -> None:
        ioc_types = set()
        for row in self.triplets:
            if row.get("isSubjectIoC") and row.get("subjectIoCType"):
                ioc_types.add(str(row["subjectIoCType"]))
            if row.get("isObjectIoC") and row.get("objectIoCType"):
                ioc_types.add(str(row["objectIoCType"]))
        for ioc_type in sorted(ioc_types):
            class_uri, class_name = self._match_ioc_type_class(mcp, ioc_type)
            self._log(logging.INFO, "[7-IoC] IoC type %s -> %s (%s)", ioc_type, class_name, class_uri)

    def _type_match_triplet(self, mcp: MCPStdioClient, row: Dict[str, Any]) -> Dict[str, Any]:
        """Match entity classes and predicate URI for one triplet.

        Thread-safe: MCP calls are serialised via MCPStdioClient._send_lock;
        cache reads/writes are protected by self._cache_lock.
        """
        subject_context = (
            f'Source sentence: {row["source_sentence"]}\n'
            f'Triplet role: subject of "{row["predicate"]}" -> "{row["object"]}"'
        )
        if row.get("subject_description"):
            subject_context = f'{subject_context}\nEntity description: {row["subject_description"]}'
        memory_summary = self._entity_memory_summary(row["subject"])
        if memory_summary:
            subject_context = f"{subject_context}\n{memory_summary}"

        if row.get("isSubjectIoC") and row.get("subjectIoCType"):
            subj_class_uri, subj_class_name = self._match_ioc_type_class(
                mcp, str(row["subjectIoCType"])
            )
            if not subj_class_uri:
                subj_class_uri, subj_class_name = self._match_entity_class(
                    mcp, row["subject"], subject_context
                )
        else:
            subj_class_uri, subj_class_name = self._match_entity_class(
                mcp, row["subject"], subject_context
            )
        object_is_literal = False
        obj_class_uri = ""
        obj_class_name = ""
        pred_uri = ""
        predicate_is_inverse = False
        object_context = (
            f'Source sentence: {row["source_sentence"]}\n'
            f'Triplet role: object of "{row["subject"]}" -> "{row["predicate"]}"'
        )
        if row.get("object_description"):
            object_context = f'{object_context}\nEntity description: {row["object_description"]}'
        object_memory_summary = self._entity_memory_summary(row["object"])
        if object_memory_summary:
            object_context = f"{object_context}\n{object_memory_summary}"

        if row["isObjectIoC"]:
            if row.get("objectIoCType"):
                obj_class_uri, obj_class_name = self._match_ioc_type_class(
                    mcp, str(row["objectIoCType"])
                )
                if not obj_class_uri:
                    obj_class_uri, obj_class_name = self._match_entity_class(
                        mcp, row["object"], object_context
                    )
            else:
                obj_class_uri, obj_class_name = self._match_entity_class(
                    mcp, row["object"], object_context
                )
            pred_uri, predicate_is_inverse, object_is_literal = self._match_predicate_uri(
                row,
                subj_class_uri,
                obj_class_uri,
            )
        else:
            obj_class_uri, obj_class_name = self._match_entity_class(
                mcp, row["object"], object_context
            )
            pred_uri, predicate_is_inverse, object_is_literal = self._match_predicate_uri(
                row,
                subj_class_uri,
                obj_class_uri,
            )
            if object_is_literal:
                obj_class_uri = ""
                obj_class_name = ""

        if object_is_literal:
            obj_class_uri = ""
            obj_class_name = ""

        typed_subject = row["subject"]
        typed_subject_description = row.get("subject_description", "")
        typed_subject_class_uri = subj_class_uri
        typed_subject_class_name = subj_class_name
        typed_object = row["object"]
        typed_object_description = row.get("object_description", "")
        typed_object_class_uri = obj_class_uri
        typed_object_class_name = obj_class_name
        typed_is_subject_ioc = row["isSubjectIoC"]
        typed_is_object_ioc = row["isObjectIoC"]
        typed_subject_ioc_type = row.get("subjectIoCType", "")
        typed_object_ioc_type = row.get("objectIoCType", "")

        if predicate_is_inverse and not object_is_literal:
            typed_subject, typed_object = typed_object, typed_subject
            typed_subject_description, typed_object_description = (
                typed_object_description,
                typed_subject_description,
            )
            typed_subject_class_uri, typed_object_class_uri = (
                typed_object_class_uri,
                typed_subject_class_uri,
            )
            typed_subject_class_name, typed_object_class_name = (
                typed_object_class_name,
                typed_subject_class_name,
            )
            typed_is_subject_ioc, typed_is_object_ioc = (
                typed_is_object_ioc,
                typed_is_subject_ioc,
            )
            typed_subject_ioc_type, typed_object_ioc_type = (
                typed_object_ioc_type,
                typed_subject_ioc_type,
            )

        typed_row = {
            "id": row["id"],
            "source_sentence": row["source_sentence"],
            "source_chunk": row["source_chunk"],
            "subject": typed_subject,
            "subject_description": typed_subject_description,
            "subject_class_uri": typed_subject_class_uri,
            "subject_class_name": typed_subject_class_name,
            "predicate": row["predicate"],
            "predicate_uri": pred_uri,
            "predicate_is_inverse": predicate_is_inverse,
            "object": typed_object,
            "object_description": typed_object_description,
            "object_is_literal": object_is_literal,
            "object_class_uri": typed_object_class_uri,
            "object_class_name": typed_object_class_name,
            "isSubjectIoC": typed_is_subject_ioc,
            "isObjectIoC": typed_is_object_ioc,
            "subjectIoCType": typed_subject_ioc_type,
            "objectIoCType": typed_object_ioc_type,
        }
        self._remember_resolved_entity_type(
            typed_subject,
            typed_subject_class_uri,
            typed_subject_class_name,
        )
        if not object_is_literal:
            self._remember_resolved_entity_type(
                typed_object,
                typed_object_class_uri,
                typed_object_class_name,
            )
        progress = self._advance_type_matching_progress()
        self._log_type_matching_result(row, typed_row, progress)
        return typed_row

    def _is_ontology_compliant_triplet(self, row: Dict[str, Any]) -> bool:
        """Keep only triplets that can be represented with ontology-backed types/properties."""
        if not row.get("subject_class_uri"):
            return False
        predicate_uri = row.get("predicate_uri", "")
        if not predicate_uri:
            return False
        if row.get("object_is_literal"):
            # Domain/range-style ontology validation is temporarily disabled because
            # valid CTI predicates are being dropped too aggressively on narrow schemas.
            # Keep the checks here commented for easy restoration later.
            #
            # return self.ontology_checker.validate_data_property(
            #     predicate_uri,
            #     row["subject_class_uri"],
            # )
            return True
        if not row.get("object_class_uri"):
            return False
        # Domain/range-style ontology validation is temporarily disabled because
        # valid CTI predicates are being dropped too aggressively on narrow schemas.
        # Keep the checks here commented for easy restoration later.
        #
        # return self.ontology_checker.validate_object_property(
        #     predicate_uri,
        #     row["subject_class_uri"],
        #     row["object_class_uri"],
        # )
        return True

    def _try_create_mcp(self) -> Optional[MCPStdioClient]:
        try:
            server_script = self._resolve_mcp_server_script()
            self._log(logging.INFO, "[7] Using MCP server script: %s", server_script)
            return MCPStdioClient(
                server_script,
                self.ontology_schema_file,
                self.logger,
                embedding_mode=str(self.config["embedding"].get("mode", "local")),
                embedding_base_url=str(self.config["embedding"].get("base_url", "")),
                embedding_model=str(self.config["embedding"].get("model", "")),
                embedding_truncate_prompt_tokens=int(
                    self.config["embedding"].get("truncate_prompt_tokens", 256)
                ),
                embedding_api_key=str(self.config["embedding"].get("api_key", "")),
                property_recommender_base_url=str(self.config["llm"].get("base_url", "")),
                property_recommender_model=str(self.config["llm"].get("model", "")),
                property_recommender_api_key=os.getenv("OPENAI_API_KEY", ""),
            )
        except Exception as e:
            self._log(logging.ERROR, "[7] MCP client creation failed: %s", e)
            return None

    def _resolve_mcp_server_script(self) -> str:
        mcp_cfg = self.config.get("mcp", {})
        extractor_root = Path(__file__).resolve().parents[1]
        candidates: List[str] = []

        configured_candidates = mcp_cfg.get("server_script_candidates", [])
        if isinstance(configured_candidates, list):
            candidates.extend(str(item) for item in configured_candidates if item)

        configured_script = mcp_cfg.get("server_script")
        if configured_script:
            candidates.append(str(configured_script))

        if not candidates:
            raise FileNotFoundError("No MCP server script configured")

        checked: List[str] = []
        for candidate in candidates:
            candidate_path = Path(candidate)
            if not candidate_path.is_absolute():
                candidate_path = (extractor_root / candidate_path).resolve()
            checked.append(str(candidate_path))
            if candidate_path.exists():
                return str(candidate_path)

        raise FileNotFoundError(
            "No configured MCP server script exists. Checked: " + ", ".join(checked)
        )

    def _verify_mcp_tools(self, mcp: MCPStdioClient) -> None:
        allowed = {
            "get_ontology_summary", "list_root_classes", "list_subclasses",
            "get_class_hierarchy", "search_classes", "search_properties",
            "get_class_details", "list_available_facets",
            "create_entity", "recommend_attribute", "recommend_relation",
        }
        tools_result = mcp.list_tools()
        available = {t.get("name") for t in tools_result.get("tools", []) if isinstance(t, dict)}
        missing = allowed - available
        if missing:
            self._log(logging.WARNING, "[7] MCP missing expected tools: %s", missing)

    def _select_root_node(self, mcp: MCPStdioClient) -> None:
        class_uri, class_name = self._match_entity_class(
            mcp,
            self.source_filename,
            "CTI threat intelligence report document",
        )
        if not class_name:
            self._log(
                logging.WARNING,
                "[7-0] Root node class could not be matched via ontology; falling back to Report"
            )
            class_name = "Report"
        self.report_root["name"] = self.source_filename
        self.report_root["class_uri"] = class_uri
        self.report_root["class_name"] = class_name

    @staticmethod
    def _parse_search_classes_result(text: str) -> List[Dict[str, str]]:
        """Parse the text output of the search_classes MCP tool into [{name, uri, description}] list."""
        candidates: List[Dict[str, str]] = []
        for block in re.split(r"\n(?=\d+\.)", text):
            name_m = re.match(r"\d+\.\s+(.+?)\s*\(Sim:", block)
            uri_m = re.search(r"URI:\s*(https?://\S+)", block)
            desc_m = re.search(r"Description:\s*(.+?)(?:\n|$)", block)
            if name_m and uri_m:
                candidates.append({
                    "name": name_m.group(1).strip(),
                    "uri": uri_m.group(1).strip().rstrip(")"),
                    "description": desc_m.group(1).strip() if desc_m else "",
                })
        return candidates

    @staticmethod
    def _parse_search_properties_result(text: str) -> List[Dict[str, str]]:
        """Parse the text output of the search_properties MCP tool into [{name, uri}] list."""
        candidates: List[Dict[str, str]] = []
        for line in text.splitlines():
            m = re.match(r"-\s+(.+?)\s+\((https?://\S+)\)\s+\[Sim:", line.strip())
            if m:
                candidates.append({
                    "name": m.group(1).strip(),
                    "uri": m.group(2).strip().rstrip(")"),
                })
        return candidates

    _QUICK_TYPE_MATCH_CONFIDENCE_THRESHOLD = 0.75

    def _pre_classify_entity(self, entity: str, context: str) -> str:
        """방안 1: 사전 카테고리 분류.

        Runs one cheap LLM call to produce a 2-4 word ontology-friendly concept label
        (e.g. "malware spyware", "organization company") that is then used as the
        search query for _quick_type_match instead of the raw entity name.

        Returns the concept string, or "" on failure.
        """
        try:
            parsed = self.llm.chat_json(
                entity_pre_classification_prompt(entity, context),
                trace_context={"event": "entity_pre_classification", "entity": entity},
            )
            if isinstance(parsed, dict):
                concept = parsed.get("concept", "").strip()
                if concept:
                    self._log(
                        logging.INFO,
                        "[pre_classify] '%s' -> '%s'",
                        entity, concept,
                    )
                    return concept
        except Exception as e:
            self._log(logging.WARNING, "[pre_classify] failed for '%s': %s", entity, e)
        return ""

    def _quick_type_match(
        self,
        mcp: MCPStdioClient,
        entity: str,
        context: str,
        search_query: str,
    ) -> Tuple[str, str]:
        """1차 패스: search_classes → type_match_select_prompt 1-shot.

        Returns (class_uri, class_name) if confidence >= threshold, else ("", "").
        Costs exactly 1 MCP tool call + 1 LLM call.
        """
        try:
            raw = self._call_mcp_tool(
                mcp, "search_classes", {"query": search_query}, "type_matching",
                trace_context={"match_kind": "quick_type_match_search", "entity": entity, "query": search_query},
            )
            candidates = self._parse_search_classes_result(raw)
            if not candidates:
                return "", ""
            parsed = self.llm.chat_json(
                type_match_select_prompt(entity, context, candidates),
                trace_context={"match_kind": "quick_type_match_select", "entity": entity, "query": search_query},
            )
            if not isinstance(parsed, dict):
                return "", ""
            class_uri = parsed.get("class_uri", "").strip()
            class_name = parsed.get("class_name", "").strip()
            confidence = float(parsed.get("confidence", 0.0))
            if confidence >= self._QUICK_TYPE_MATCH_CONFIDENCE_THRESHOLD and class_uri:
                is_valid, _ = self._validate_class_uri(class_uri)
                if is_valid:
                    self._log(
                        logging.INFO,
                        "[quick_type_match] '%s' -> '%s' (%s) conf=%.2f query='%s'",
                        entity, class_name, class_uri, confidence, search_query,
                    )
                    return class_uri, class_name
        except Exception as e:
            self._log(logging.WARNING, "[quick_type_match] failed for '%s': %s", entity, e)
        return "", ""

    def _match_entity_class(
        self, mcp: MCPStdioClient, entity: str, context: str
    ) -> Tuple[str, str]:
        memory_summary = self._entity_memory_summary(entity)
        best_evidence = ""
        if memory_summary:
            best_evidence = memory_summary.splitlines()[0]
        cache_key = f"{entity.strip().lower()}||{best_evidence.strip().lower()}"
        with self._cache_lock:
            if cache_key in self._entity_class_cache:
                return self._entity_class_cache[cache_key]

        effective_context = context
        if memory_summary and memory_summary not in effective_context:
            effective_context = f"{effective_context}\n{memory_summary}"

        # 방안 1: pre-classify entity to get an ontology-friendly concept hint
        concept_hint = self._pre_classify_entity(entity, effective_context)

        # 방안 4: 1차 패스 — search_classes + type_match_select_prompt (1 MCP call)
        # Try with the concept hint first; if that fails, fall back to raw entity name.
        search_query = concept_hint if concept_hint else entity.strip()
        class_uri, class_name = self._quick_type_match(mcp, entity, effective_context, search_query)
        if not class_uri and concept_hint:
            # concept hint did not yield high-confidence result — retry with entity name directly
            class_uri, class_name = self._quick_type_match(mcp, entity, effective_context, entity.strip())
        if class_uri:
            result = (class_uri, class_name)
            with self._cache_lock:
                self._entity_class_cache[cache_key] = result
            return result

        # Fall through: full MCP agent loop
        # Inject concept hint into context so the agent starts with a directional hint
        agent_context = (
            f"{effective_context}\n[Pre-classified concept: {concept_hint}]"
            if concept_hint else effective_context
        )

        max_per_entity = int(self.config["mcp"]["max_tool_calls_type_matching"])
        allowed_tools = {
            "search_classes",
            "list_root_classes",
            "list_subclasses",
            "get_class_hierarchy",
            "get_class_details",
            "list_available_facets",
            "drill_into_classes",
        }
        validation_feedback = ""
        class_uri = ""
        class_name = ""
        for _ in range(3):
            parsed = self._run_mcp_agent_loop(
                mcp=mcp,
                prompt_factory=lambda transcript, remaining, force_finish=False, feedback=validation_feedback, _ctx=agent_context: class_resolution_agent_prompt(
                    entity=entity,
                    context=_ctx,
                    transcript=self._augment_transcript(transcript, feedback),
                    remaining_calls=remaining,
                    force_finish=force_finish,
                ),
                allowed_tools=allowed_tools,
                max_calls=max_per_entity,
                counter_key="type_matching",
                trace_context={
                    "match_kind": "class_type_matching",
                    "entity": entity,
                    "context": agent_context,
                },
            )
            class_uri = parsed.get("class_uri", "") if isinstance(parsed, dict) else ""
            class_name = parsed.get("class_name", "") if isinstance(parsed, dict) else ""
            if class_uri and not self._response_uri_seen(class_uri, parsed.get("_transcript", "")):
                class_uri = ""
                class_name = ""

            is_valid, reason = self._validate_class_uri(class_uri)
            if is_valid:
                break

            validation_feedback = (
                f"{reason} The previous answer is invalid. "
                "Return a class URI that actually exists in the loaded ontology, or return an empty class URI if no valid class fits."
            )
            self._log(logging.WARNING, "[MCP-Agent][type_matching] %s", reason)
            class_uri = ""
            class_name = ""

        result = (class_uri, class_name)
        with self._cache_lock:
            self._entity_class_cache[cache_key] = result
        return result

    def _check_is_literal(
        self,
        subject: str,
        row_id: Any,
        subject_class_uri: str,
        predicate: str,
        obj: str,
    ) -> Tuple[bool, str]:
        if not subject_class_uri:
            return False, ""

        cache_key = (subject_class_uri, predicate)
        with self._cache_lock:
            if cache_key in self._literal_check_cache:
                return self._literal_check_cache[cache_key]

        max_calls = int(self.config["mcp"]["max_tool_calls_property_matching"])
        validation_feedback = ""
        result = (False, "")
        for _ in range(3):
            parsed = self._run_property_agent_with_temp_entities(
                subject_class_uri=subject_class_uri,
                object_class_uri="",
                prompt_factory=lambda transcript, remaining, temp_subject_uri, temp_object_uri, force_finish=False, feedback=validation_feedback: data_property_resolution_agent_prompt(
                    subject=subject,
                    subject_class_uri=subject_class_uri,
                    subject_entity_uri=temp_subject_uri,
                    predicate=predicate,
                    obj=obj,
                    transcript=self._augment_transcript(transcript, feedback),
                    remaining_calls=remaining,
                    force_finish=force_finish,
                ),
                allowed_tools={"recommend_attribute", "search_properties", "get_class_details", "list_available_facets"},
                max_calls=max_calls,
                counter_key="property_matching",
                subject_entity_id=self._temp_mcp_entity_id(row_id, "subject"),
                trace_context={
                    "match_kind": "relationship_type_matching",
                    "property_mode": "data_property",
                    "row_id": row_id,
                    "subject": subject,
                    "subject_class_uri": subject_class_uri,
                    "predicate": predicate,
                    "object": obj,
                },
            )
            prop_uri = parsed.get("property_uri", "") if isinstance(parsed, dict) else ""
            if prop_uri and not self._response_uri_seen(prop_uri, parsed.get("_transcript", "")):
                prop_uri = ""
            is_data_property = isinstance(parsed, dict) and bool(parsed.get("is_data_property", False))
            is_valid, reason = self._validate_property_uri(
                prop_uri,
                expect_data_property=True if is_data_property else None,
            )
            if not is_valid:
                validation_feedback = (
                    f"{reason} The previous answer is invalid. "
                    "Return a datatype property URI that actually exists in the loaded ontology, or return an empty property URI if this should not be a data property."
                )
                self._log(logging.WARNING, "[MCP-Agent][property_matching] %s", reason)
                continue
            if is_data_property and prop_uri:
                result = (True, prop_uri)
            else:
                result = (False, "")
            break
        with self._cache_lock:
            self._literal_check_cache[cache_key] = result
        return result

    def _predicate_query_variants(
        self,
        subject: str,
        predicate: str,
        obj: str,
        subject_class_uri: str,
        obj_class_uri: str,
    ) -> List[str]:
        cache_key = (
            subject_class_uri or "",
            predicate.strip().casefold(),
            obj_class_uri or "",
        )
        with self._cache_lock:
            cached = self._predicate_query_cache.get(cache_key)
        if cached is not None:
            return cached

        variants: List[str] = []
        base = predicate.strip()
        if base:
            variants.append(base)

        try:
            parsed = self.llm.chat_json(
                predicate_query_expansion_prompt(
                    subject=subject,
                    predicate=predicate,
                    obj=obj,
                    subject_class_uri=subject_class_uri,
                    object_class_uri=obj_class_uri,
                ),
                trace_context={
                    "event": "predicate_query_expansion",
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "subject_class_uri": subject_class_uri,
                    "object_class_uri": obj_class_uri,
                },
            )
            queries = parsed.get("queries", []) if isinstance(parsed, dict) else []
            if isinstance(queries, list):
                for item in queries:
                    if isinstance(item, str) and item.strip():
                        variants.append(item.strip())
        except Exception as e:
            self._log(
                logging.WARNING,
                "[MCP-Agent][property_matching] predicate query expansion failed for '%s': %s",
                predicate,
                e,
            )

        normalized: List[str] = []
        seen: set[str] = set()
        for item in variants:
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(item)
            if len(normalized) >= 5:
                break
        if not normalized and base:
            normalized = [base]

        with self._cache_lock:
            self._predicate_query_cache[cache_key] = normalized
        return normalized

    def _should_attempt_object_property_first(self, row: Dict[str, Any]) -> bool:
        if row.get("isObjectIoC"):
            return True
        if row.get("object_description"):
            return True

        chunk_name = str(row.get("source_chunk", "")).strip()
        object_name = str(row.get("object", "")).strip()
        if not chunk_name or not object_name:
            return False

        target = self._normalize_entity_key(object_name)
        if not target:
            return False
        for item in self.entities_by_chunk.get(chunk_name, []):
            if self._normalize_entity_key(item.get("name", "")) == target:
                return True
        return False

    def _recommend_property_candidates(
        self,
        row: Dict[str, Any],
        subject_class_uri: str,
        object_class_uri: str,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        mcp = self._try_create_mcp()
        if mcp is None:
            return False, []

        try:
            with mcp:
                payload = self._call_mcp_tool_structured(
                    mcp,
                    "recommend_property",
                    {
                        "subject_type_uri": subject_class_uri,
                        "object_type_uri": object_class_uri,
                        "predicate": row["predicate"],
                        "context": row["source_sentence"],
                    },
                    "property_matching",
                    trace_context={
                        "match_kind": "relationship_type_matching",
                        "property_mode": "recommend_property",
                        "row_id": row["id"],
                        "subject": row["subject"],
                        "subject_class_uri": subject_class_uri,
                        "predicate": row["predicate"],
                        "object": row["object"],
                        "object_class_uri": object_class_uri,
                    },
                )
        except Exception as e:
            self._log(
                logging.WARNING,
                "[predicate_matching] recommend_property failed for row %s: %s",
                row.get("id"),
                e,
            )
            return False, []

        if not isinstance(payload, dict):
            self._log(
                logging.WARNING,
                "[predicate_matching] recommend_property returned non-dict payload for row %s",
                row.get("id"),
            )
            return False, []

        if not bool(payload.get("isSuccess", False)):
            self._log(
                logging.WARNING,
                "[predicate_matching] MCP recommend_property failed for row %s",
                row.get("id"),
            )
            return False, []

        candidates: List[Dict[str, Any]] = []
        for item in payload.get("result", []) or []:
            if not isinstance(item, dict):
                continue
            property_uri = str(item.get("propertyURI", "") or "").strip()
            if not property_uri:
                continue
            candidates.append({
                "property_uri": property_uri,
                "property_name": self._short_uri_label(property_uri),
                "is_inverse": bool(item.get("isReverse", False)),
                "is_data_property": bool(item.get("isDataProperty", False)),
                "confidence": float(item.get("confidence", 0.0) or 0.0),
                "source": "recommend_property",
                "note": "Candidate suggested by MCP recommend_property.",
            })
        return True, candidates

    def _recommend_attribute_candidates(
        self,
        row: Dict[str, Any],
        subject_class_uri: str,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        mcp = self._try_create_mcp()
        if mcp is None:
            return False, []

        subject_entity_id = self._temp_mcp_entity_id(row["id"], "subject")
        temp_subject_uri = self._temp_mcp_entity_uri(subject_entity_id)
        trace_context = {
            "match_kind": "relationship_type_matching",
            "property_mode": "recommend_attribute",
            "row_id": row["id"],
            "subject": row["subject"],
            "subject_class_uri": subject_class_uri,
            "predicate": row["predicate"],
            "object": row["object"],
            "subject_entity_id": subject_entity_id,
            "subject_entity_uri": temp_subject_uri,
        }

        try:
            with mcp:
                self._call_mcp_tool(
                    mcp,
                    "create_entity",
                    {"entity_id": subject_entity_id, "class_uris": [subject_class_uri]},
                    "property_matching",
                    trace_context=trace_context,
                )
                raw = self._call_mcp_tool(
                    mcp,
                    "recommend_attribute",
                    {
                        "entity_uri": temp_subject_uri,
                        "query": row["predicate"],
                        "value": row["object"],
                        "context": row["source_sentence"],
                    },
                    "property_matching",
                    trace_context=trace_context,
                )
        except Exception as e:
            self._log(
                logging.WARNING,
                "[predicate_matching] recommend_attribute failed for row %s: %s",
                row.get("id"),
                e,
            )
            return False, []

        return True, self._parse_ranked_property_recommendations(raw)

    def _match_predicate_uri(
        self,
        row: Dict[str, Any],
        subject_class_uri: str,
        object_class_uri: str,
    ) -> Tuple[str, bool, bool]:
        cache_key = (
            subject_class_uri,
            row["predicate"],
            object_class_uri,
            str(row.get("source_chunk", "") or ""),
            str(row.get("object", "") or ""),
        )
        with self._cache_lock:
            if cache_key in self._predicate_uri_cache:
                return self._predicate_uri_cache[cache_key]

        result = ""
        is_inverse = False
        is_data_property = False

        if not subject_class_uri:
            final = (result, is_inverse, is_data_property)
            with self._cache_lock:
                self._predicate_uri_cache[cache_key] = final
            return final

        if object_class_uri:
            ok, candidates = self._recommend_property_candidates(row, subject_class_uri, object_class_uri)
            if ok and candidates:
                result, is_inverse, is_data_property = self._judge_predicate_candidates(
                    row=row,
                    subject_class_uri=subject_class_uri,
                    object_class_uri=object_class_uri,
                    candidates=candidates,
                )
        else:
            ok, candidates = self._recommend_attribute_candidates(row, subject_class_uri)
            if ok and candidates:
                result, is_inverse, is_data_property = self._judge_predicate_candidates(
                    row=row,
                    subject_class_uri=subject_class_uri,
                    object_class_uri="",
                    candidates=candidates,
                )

        expect_data_property = is_data_property if result else None
        is_valid, reason = self._validate_property_uri(result, expect_data_property=expect_data_property)
        if not is_valid:
            self._log(logging.WARNING, "[predicate_matching] %s", reason)
            result = ""
            is_inverse = False
            is_data_property = False

        final = (result, is_inverse, is_data_property)
        with self._cache_lock:
            self._predicate_uri_cache[cache_key] = final
        return final

    def _run_mcp_agent_loop(
        self,
        mcp: MCPStdioClient,
        prompt_factory: Any,
        allowed_tools: set,
        max_calls: int,
        counter_key: str,
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        transcript_entries: List[Dict[str, Any]] = []
        session_id = uuid4().hex
        base_trace = {
            "agent_session_id": session_id,
            "counter_key": counter_key,
            "allowed_tools": sorted(allowed_tools),
            "max_calls": max_calls,
            **(trace_context or {}),
        }
        self._trace_event("agent_session_start", **base_trace)
        for call_index in range(max_calls):
            remaining = max_calls - call_index
            prompt = prompt_factory(self._format_agent_transcript(transcript_entries), remaining, False)
            step_trace = {
                **base_trace,
                "step": call_index + 1,
                "remaining_calls": remaining,
                "prompt": prompt,
                "transcript_before_step": self._format_agent_transcript(transcript_entries),
            }
            self._trace_event("agent_prompt", **step_trace)
            self._log(
                logging.INFO,
                "[MCP-Agent][%s][step %s/%s] Prompt:\n%s",
                counter_key,
                call_index + 1,
                max_calls,
                self._truncate_for_log(prompt),
            )
            parsed = self.llm.chat_json(
                prompt,
                trace_context={
                    **base_trace,
                    "step": call_index + 1,
                    "remaining_calls": remaining,
                },
            )
            self._trace_event(
                "agent_llm_parsed",
                **base_trace,
                step=call_index + 1,
                remaining_calls=remaining,
                parsed=parsed,
            )
            self._log(
                logging.INFO,
                "[MCP-Agent][%s][step %s/%s] LLM response: %s",
                counter_key,
                call_index + 1,
                max_calls,
                self._truncate_for_log(json.dumps(parsed, ensure_ascii=False)),
            )
            if not isinstance(parsed, dict):
                break
            action = str(parsed.get("action", "finish")).strip().lower()
            if action == "finish":
                parsed["_transcript"] = self._format_agent_transcript(transcript_entries)
                self._trace_event(
                    "agent_finish",
                    **base_trace,
                    step=call_index + 1,
                    remaining_calls=remaining,
                    parsed=parsed,
                )
                self._trace_event(
                    "agent_session_end",
                    **base_trace,
                    final_response=parsed,
                    transcript=parsed["_transcript"],
                )
                return parsed
            if action != "call_tool":
                self._trace_event(
                    "agent_invalid_action",
                    **base_trace,
                    step=call_index + 1,
                    remaining_calls=remaining,
                    parsed=parsed,
                )
                break

            tool = str(parsed.get("tool", "")).strip()
            if tool not in allowed_tools:
                self._trace_event(
                    "agent_tool_rejected",
                    **base_trace,
                    step=call_index + 1,
                    remaining_calls=remaining,
                    tool=tool or "<invalid>",
                    arguments=parsed.get("arguments", {}),
                )
                transcript_entries.append({
                    "tool": tool or "<invalid>",
                    "arguments": parsed.get("arguments", {}),
                    "result": f"ERROR: Tool not allowed. Allowed tools: {sorted(allowed_tools)}",
                })
                continue

            arguments = parsed.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            try:
                with self._mcp_count_lock:
                    self.mcp_call_count[counter_key] += 1
                self._trace_event(
                    "agent_tool_call",
                    **base_trace,
                    step=call_index + 1,
                    remaining_calls=remaining,
                    tool=tool,
                    arguments=arguments,
                )
                self._log(
                    logging.INFO,
                    "[MCP-Agent][%s][step %s/%s] Tool call: %s %s",
                    counter_key,
                    call_index + 1,
                    max_calls,
                    tool,
                    self._truncate_for_log(json.dumps(arguments, ensure_ascii=False)),
                )
                result = mcp.call_tool(tool, arguments)
            except Exception as e:
                result = f"ERROR: {e}"
            self._trace_event(
                "agent_tool_result",
                **base_trace,
                step=call_index + 1,
                remaining_calls=remaining,
                tool=tool,
                arguments=arguments,
                result=result,
            )
            self._log(
                logging.INFO,
                "[MCP-Agent][%s][step %s/%s] Tool result from %s:\n%s",
                counter_key,
                call_index + 1,
                max_calls,
                tool,
                self._truncate_for_log(result),
            )
            transcript_entries.append({
                "tool": tool,
                "arguments": arguments,
                "result": result,
            })

        final_prompt = prompt_factory(self._format_agent_transcript(transcript_entries), 0, True)
        self._trace_event(
            "agent_final_prompt",
            **base_trace,
            step=max_calls + 1,
            remaining_calls=0,
            prompt=final_prompt,
            transcript_before_step=self._format_agent_transcript(transcript_entries),
        )
        self._log(
            logging.INFO,
            "[MCP-Agent][%s][final] Prompt:\n%s",
            counter_key,
            self._truncate_for_log(final_prompt),
        )
        parsed = self.llm.chat_json(
            final_prompt,
            trace_context={
                **base_trace,
                "step": max_calls + 1,
                "remaining_calls": 0,
                "phase": "final",
            },
        )
        self._trace_event(
            "agent_final_llm_parsed",
            **base_trace,
            step=max_calls + 1,
            remaining_calls=0,
            parsed=parsed,
        )
        self._log(
            logging.INFO,
            "[MCP-Agent][%s][final] LLM response: %s",
            counter_key,
            self._truncate_for_log(json.dumps(parsed, ensure_ascii=False)),
        )
        if not isinstance(parsed, dict):
            parsed = {}
        parsed["_transcript"] = self._format_agent_transcript(transcript_entries)
        self._trace_event(
            "agent_session_end",
            **base_trace,
            final_response=parsed,
            transcript=parsed["_transcript"],
        )
        return parsed

    @staticmethod
    def _temp_mcp_entity_id(row_id: Any, role: str) -> str:
        return f"row_{row_id}_{role}"

    @staticmethod
    def _temp_mcp_entity_uri(entity_id: str) -> str:
        return f"http://example.org/entities/{quote(str(entity_id).replace(' ', '_'))}"

    def _call_mcp_tool(
        self,
        mcp: MCPStdioClient,
        tool: str,
        arguments: Dict[str, Any],
        counter_key: str,
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        with self._mcp_count_lock:
            self.mcp_call_count[counter_key] += 1
        if trace_context:
            self._trace_event(
                "agent_setup_tool_call",
                counter_key=counter_key,
                tool=tool,
                arguments=arguments,
                **trace_context,
            )
        self._log(
            logging.INFO,
            "[MCP-Agent][%s][setup] Tool call: %s %s",
            counter_key,
            tool,
            self._truncate_for_log(json.dumps(arguments, ensure_ascii=False)),
        )
        result = mcp.call_tool(tool, arguments)
        if trace_context:
            self._trace_event(
                "agent_setup_tool_result",
                counter_key=counter_key,
                tool=tool,
                arguments=arguments,
                result=result,
                **trace_context,
            )
        self._log(
            logging.INFO,
            "[MCP-Agent][%s][setup] Tool result from %s:\n%s",
            counter_key,
            tool,
            self._truncate_for_log(result),
        )
        return result

    def _call_mcp_tool_structured(
        self,
        mcp: MCPStdioClient,
        tool: str,
        arguments: Dict[str, Any],
        counter_key: str,
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        with self._mcp_count_lock:
            self.mcp_call_count[counter_key] += 1
        if trace_context:
            self._trace_event(
                "agent_setup_tool_call",
                counter_key=counter_key,
                tool=tool,
                arguments=arguments,
                **trace_context,
            )
        result = mcp.call_tool_structured(tool, arguments)
        if trace_context:
            self._trace_event(
                "agent_setup_tool_result",
                counter_key=counter_key,
                tool=tool,
                arguments=arguments,
                result=result,
                **trace_context,
            )
        self._log(
            logging.INFO,
            "[MCP-Agent][%s][setup] Structured tool result from %s:\n%s",
            counter_key,
            tool,
            self._truncate_for_log(json.dumps(result, ensure_ascii=False)),
        )
        return result

    def _run_property_agent_with_temp_entities(
        self,
        subject_class_uri: str,
        object_class_uri: str,
        prompt_factory: Any,
        allowed_tools: set,
        max_calls: int,
        counter_key: str,
        subject_entity_id: str,
        object_entity_id: str = "",
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not subject_class_uri:
            return {}
        if object_entity_id and not object_class_uri:
            return {}
        mcp = self._try_create_mcp()
        if mcp is None:
            return {}

        temp_subject_uri = self._temp_mcp_entity_uri(subject_entity_id)
        temp_object_uri = self._temp_mcp_entity_uri(object_entity_id) if object_entity_id else ""
        setup_trace = {
            "subject_entity_id": subject_entity_id,
            "subject_entity_uri": temp_subject_uri,
            "object_entity_id": object_entity_id,
            "object_entity_uri": temp_object_uri,
            **(trace_context or {}),
        }
        try:
            with mcp:
                self._call_mcp_tool(
                    mcp,
                    "create_entity",
                    {"entity_id": subject_entity_id, "class_uris": [subject_class_uri]},
                    counter_key,
                    trace_context=setup_trace,
                )
                if object_entity_id and object_class_uri:
                    self._call_mcp_tool(
                        mcp,
                        "create_entity",
                        {"entity_id": object_entity_id, "class_uris": [object_class_uri]},
                        counter_key,
                        trace_context=setup_trace,
                    )
                return self._run_mcp_agent_loop(
                    mcp=mcp,
                    prompt_factory=lambda transcript, remaining, force_finish=False: prompt_factory(
                        transcript,
                        remaining,
                        temp_subject_uri,
                        temp_object_uri,
                        force_finish,
                    ),
                    allowed_tools=allowed_tools,
                    max_calls=max_calls,
                    counter_key=counter_key,
                    trace_context=setup_trace,
                )
        except Exception as e:
            self._log(logging.ERROR, "[MCP-Agent][%s] temporary graph setup failed: %s", counter_key, e)
            return {}

    @staticmethod
    def _format_agent_transcript(entries: List[Dict[str, Any]], max_chars: int = 12000) -> str:
        if not entries:
            return "(no tool calls yet)"
        parts: List[str] = []
        for i, entry in enumerate(entries, start=1):
            parts.append(
                f"{i}. TOOL: {entry.get('tool')}\n"
                f"   ARGUMENTS: {json.dumps(entry.get('arguments', {}), ensure_ascii=False)}\n"
                f"   RESULT:\n{str(entry.get('result', ''))}"
            )
        text = "\n".join(parts)
        if len(text) > max_chars:
            return text[-max_chars:]
        return text

    @staticmethod
    def _response_uri_seen(uri: str, transcript: str) -> bool:
        return bool(uri) and uri in transcript

    @staticmethod
    def _augment_transcript(transcript: str, validation_feedback: str) -> str:
        if not validation_feedback:
            return transcript
        return f"{transcript}\n\nLOCAL VALIDATION FEEDBACK:\n{validation_feedback}"

    @staticmethod
    def _looks_like_real_uri(uri: str) -> bool:
        if not uri or not uri.startswith(("http://", "https://")):
            return False
        if " " in uri or "->" in uri:
            return False
        if uri.startswith("http://example.org/entities/"):
            return False
        return True

    def _validate_class_uri(self, class_uri: str) -> Tuple[bool, str]:
        if not class_uri:
            return True, ""
        if not self._looks_like_real_uri(class_uri):
            return False, f"Class URI '{class_uri}' is not a real ontology class URI."
        if self.ontology_checker.available and not self.ontology_checker.has_class_uri(class_uri):
            return False, f"Class URI '{class_uri}' does not exist in the loaded ontology schema."
        return True, ""

    def _validate_property_uri(
        self,
        property_uri: str,
        expect_data_property: Optional[bool] = None,
    ) -> Tuple[bool, str]:
        if not property_uri:
            return True, ""
        if not self._looks_like_real_uri(property_uri):
            return False, f"Property URI '{property_uri}' is not a real ontology property URI."
        if not self.ontology_checker.available:
            return True, ""
        if expect_data_property is True:
            if not self.ontology_checker.has_data_property_uri(property_uri):
                return False, f"Property URI '{property_uri}' is not a datatype property in the loaded ontology schema."
            return True, ""
        if expect_data_property is False:
            if not self.ontology_checker.has_object_property_uri(property_uri):
                return False, f"Property URI '{property_uri}' is not an object property in the loaded ontology schema."
            return True, ""
        if self.ontology_checker.has_object_property_uri(property_uri) or self.ontology_checker.has_data_property_uri(property_uri):
            return True, ""
        return False, f"Property URI '{property_uri}' does not exist in the loaded ontology schema."

    def _encode_name_map(self, names: List[str]) -> Dict[str, List[float]]:
        unique_names = []
        seen: set[str] = set()
        for name in names:
            if name and name not in seen:
                seen.add(name)
                unique_names.append(name)
        vectors = self.embedding.encode_many(unique_names)
        return {name: vec for name, vec in zip(unique_names, vectors)}

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # MCP result parsers
    # ------------------------------------------------------------------

    def _parse_class_list(self, text: str) -> List[Dict[str, str]]:
        results = []
        for line in text.splitlines():
            m = re.search(r"-\s*(.+?)\s*\((https?://[^\s)]+)\)", line)
            if m:
                results.append({"name": m.group(1).strip(), "uri": m.group(2).strip()})
                continue
            for uri in re.findall(r"https?://[^\s)]+", line):
                name = re.sub(r"\(.*$", "", line.lstrip("- ")).strip()
                if name:
                    results.append({"name": name, "uri": uri})
        return results

    def _parse_property_list(self, text: str) -> List[Dict[str, str]]:
        props = []
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            m = re.search(r"-\s*(.+?)\s*\((https?://[^\s)]+)\)", lines[i])
            if m:
                prop: Dict[str, str] = {"name": m.group(1).strip(), "uri": m.group(2).strip(), "type": ""}
                j = i + 1
                while j < len(lines) and lines[j].startswith("  "):
                    m2 = re.search(r"Type:\s*([A-Za-z]+)", lines[j])
                    if m2:
                        prop["type"] = m2.group(1).strip()
                    j += 1
                props.append(prop)
                i = j
            else:
                i += 1
        return props

    @staticmethod
    def _short_uri_label(uri: str) -> str:
        if not uri:
            return ""
        uri = uri.rstrip("/")
        if "#" in uri:
            return uri.rsplit("#", 1)[-1]
        return uri.rsplit("/", 1)[-1]

    def _get_chunk_text(self, chunk_name: str) -> str:
        chunk_key = str(chunk_name or "").strip()
        if not chunk_key:
            return ""
        with self._cache_lock:
            cached = self._chunk_text_cache.get(chunk_key)
        if cached is not None:
            return cached

        text = ""
        for path in self.chunk_files:
            if path.stem == chunk_key:
                text = self._read_text(path)
                break
        with self._cache_lock:
            self._chunk_text_cache[chunk_key] = text
        return text

    @staticmethod
    def _parse_ranked_property_recommendations(text: str) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for block in re.split(r"\n(?=\d+\.)", text):
            match = re.search(
                r"^\s*\d+\.\s+(.+?)\s+\((https?://\S+)\)\s+\[Score:\s*([0-9.]+)\]",
                block,
                re.MULTILINE,
            )
            if not match:
                continue
            candidates.append({
                "property_name": match.group(1).strip(),
                "property_uri": match.group(2).strip().rstrip(")"),
                "confidence": float(match.group(3)),
                "is_inverse": False,
                "is_data_property": True,
                "source": "recommend_attribute",
                "note": "DatatypeProperty candidate suggested by MCP recommend_attribute.",
            })
        return candidates

    @staticmethod
    def _normalize_property_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        seen: set[Tuple[str, bool, bool]] = set()
        for item in candidates:
            property_uri = str(item.get("property_uri", "") or "").strip()
            if not property_uri:
                continue
            is_inverse = bool(item.get("is_inverse", False))
            is_data_property = bool(item.get("is_data_property", False))
            if is_inverse and is_data_property:
                continue
            key = (property_uri, is_inverse, is_data_property)
            if key in seen:
                continue
            seen.add(key)
            normalized.append({
                "property_uri": property_uri,
                "property_name": str(item.get("property_name", "") or "").strip(),
                "is_inverse": is_inverse,
                "is_data_property": is_data_property,
                "confidence": float(item.get("confidence", 0.0) or 0.0),
                "source": str(item.get("source", "") or "").strip(),
                "note": str(item.get("note", "") or "").strip(),
            })
        return normalized

    def _judge_predicate_candidates(
        self,
        row: Dict[str, Any],
        subject_class_uri: str,
        object_class_uri: str,
        candidates: List[Dict[str, Any]],
    ) -> Tuple[str, bool, bool]:
        normalized = self._normalize_property_candidates(candidates)
        if not normalized:
            return "", False, False

        chunk_text = self._get_chunk_text(str(row.get("source_chunk", "")))
        parsed = self.llm.chat_json(
            predicate_candidate_judge_prompt(
                chunk_text=chunk_text,
                sentence=str(row.get("source_sentence", "") or ""),
                subject=str(row.get("subject", "") or ""),
                subject_class_uri=subject_class_uri,
                predicate=str(row.get("predicate", "") or ""),
                obj=str(row.get("object", "") or ""),
                object_class_uri=object_class_uri,
                candidates=normalized,
            ),
            trace_context={
                "match_kind": "predicate_candidate_judge",
                "row_id": row.get("id"),
                "predicate": row.get("predicate", ""),
                "source_chunk": row.get("source_chunk", ""),
            },
        )
        if not isinstance(parsed, dict):
            return "", False, False

        property_uri = str(parsed.get("property_uri", "") or "").strip()
        if not property_uri:
            return "", False, False

        selected = next((item for item in normalized if item["property_uri"] == property_uri), None)
        if selected is None:
            self._log(
                logging.WARNING,
                "[predicate_judge] rejected non-candidate property '%s' for row %s",
                property_uri,
                row.get("id"),
            )
            return "", False, False

        return (
            property_uri,
            bool(selected["is_inverse"]),
            bool(selected["is_data_property"]),
        )

    # ------------------------------------------------------------------
    # Node 8: Internal Entity Resolution
    # ------------------------------------------------------------------

    def node_internal_entity_resolution(self) -> None:
        self._log(logging.INFO, "[8] Internal Entity Resolution start")
        merged = 0
        min_similarity = float(self.config["entity_resolution"].get("min_similarity", 0.6))

        for is_obj in (False, True):
            groups: Dict[str, List[str]] = {}
            for trip in self.typed_triplets:
                if is_obj and trip["object_is_literal"]:
                    continue
                if is_obj and trip["isObjectIoC"]:
                    continue
                if (not is_obj) and trip["isSubjectIoC"]:
                    continue
                class_uri = trip["object_class_uri"] if is_obj else trip["subject_class_uri"]
                if not class_uri:
                    continue
                if class_uri.endswith("#Unknown"):
                    continue
                name = trip["object"] if is_obj else trip["subject"]
                if name not in groups.setdefault(class_uri, []):
                    groups[class_uri].append(name)

            for class_uri, names in groups.items():
                embeddings = self._encode_name_map(names)
                i = 0
                while i < len(names):
                    j = i + 1
                    while j < len(names):
                        a, b = names[i], names[j]
                        if a == b:
                            j += 1
                            continue
                        if self._cosine_similarity(embeddings.get(a, []), embeddings.get(b, [])) < min_similarity:
                            j += 1
                            continue
                        parsed = self.llm.chat_json(entity_resolution_prompt(a, b, class_uri))
                        if (
                            isinstance(parsed, dict)
                            and parsed.get("is_same") is True
                            and parsed.get("canonical_name")
                        ):
                            canonical = parsed["canonical_name"]
                            # Replace ALL references to either a or b with canonical
                            for trip in self.typed_triplets:
                                field = "object" if is_obj else "subject"
                                if trip[field] in (a, b):
                                    trip[field] = canonical
                            # Update names list to avoid stale comparisons
                            for idx_n, n in enumerate(names):
                                if n in (a, b):
                                    names[idx_n] = canonical
                            if canonical not in embeddings:
                                embeddings[canonical] = embeddings.get(a) or embeddings.get(b) or self.embedding.encode(canonical)
                            self._log(
                                logging.INFO,
                                '[Entity Resolution] Merged "%s" + "%s" -> "%s" (Type: %s)',
                                a, b, canonical, class_uri,
                            )
                            merged += 1
                            # Don't increment j; re-check position i with new canonical
                        else:
                            j += 1
                    i += 1

        self._write_json(self.intermediate / "typedTriplets.json", self.all_typed_triplets)
        self._log(logging.INFO, "[8] Internal Entity Resolution completed. merged=%s", merged)

    # ------------------------------------------------------------------
    # Node 9: Existing Entity Resolution
    # ------------------------------------------------------------------

    def node_existing_entity_resolution(self) -> None:
        self._log(logging.INFO, "[9] Existing Entity Resolution start")
        top_k = int(self.config["entity_resolution"]["top_k"])
        min_similarity = float(self.config["entity_resolution"].get("min_similarity", 0.6))
        merged = 0

        # Collect unique (name, class_uri) pairs
        seen: set = set()
        unique: List[Tuple[str, str]] = []
        for trip in self.typed_triplets:
            for name, class_uri in (
                (trip["subject"], trip["subject_class_uri"]),
                (trip["object"], trip["object_class_uri"]),
            ):
                if (name, class_uri) == (trip["subject"], trip["subject_class_uri"]) and trip["isSubjectIoC"]:
                    continue
                if (name, class_uri) == (trip["object"], trip["object_class_uri"]) and trip["isObjectIoC"]:
                    continue
                if trip.get("object_is_literal") and (name, class_uri) == (trip["object"], trip["object_class_uri"]):
                    continue
                if class_uri.endswith("#Unknown"):
                    continue
                if name and class_uri and (name, class_uri) not in seen:
                    seen.add((name, class_uri))
                    unique.append((name, class_uri))

        # Pre-encode all unique names in one batch to avoid redundant embedding calls
        unique_name_list = list({name for name, _ in unique if name})
        name_vec_cache: Dict[str, List[float]] = {}
        try:
            vecs = self.embedding.encode_many(unique_name_list)
            name_vec_cache = {n: v for n, v in zip(unique_name_list, vecs)}
        except Exception as e:
            self._log(logging.WARNING, "[9] Batch embedding failed, will fall back per-name: %s", e)

        for name, class_uri in unique:
            vec = name_vec_cache.get(name)
            if vec is None:
                try:
                    vec = self.vector_store.encode(name)
                    name_vec_cache[name] = vec
                except Exception as e:
                    self._log(logging.WARNING, "[9] Embedding failed for %s: %s", name, e)
                    continue

            hits = self.vector_store.query(vec, class_uri, topk=top_k)
            for hit in hits:
                existing_name = hit["name"]
                if existing_name == name:
                    continue
                if float(hit.get("score", 0.0)) < min_similarity:
                    continue
                parsed = self.llm.chat_json(entity_resolution_prompt(name, existing_name, class_uri))
                if isinstance(parsed, dict) and parsed.get("is_same") is True:
                    # Use existing Neo4j entity name as canonical
                    canonical = existing_name
                    for trip in self.typed_triplets:
                        if trip["subject"] == name and trip["subject_class_uri"] == class_uri:
                            trip["subject"] = canonical
                        if (
                            not trip["object_is_literal"]
                            and trip["object"] == name
                            and trip["object_class_uri"] == class_uri
                        ):
                            trip["object"] = canonical
                    self._log(
                        logging.INFO,
                        '[Existing Entity Resolution] Merged "%s" -> existing "%s" (Type: %s)',
                        name, canonical, class_uri,
                    )
                    merged += 1
                    break

        self._write_json(self.intermediate / "typedTriplets.json", self.all_typed_triplets)
        self._log(logging.INFO, "[9] Existing Entity Resolution completed. merged=%s", merged)

    # ------------------------------------------------------------------
    # Node 10: Data Insert (stub — Neo4j removed; eval uses in-memory results only)
    # ------------------------------------------------------------------

    def node_data_insert(self) -> None:
        self._log(logging.INFO, "[10] Data Insert skipped (Neo4j removed)")

    # ------------------------------------------------------------------
    # LangGraph orchestration
    # ------------------------------------------------------------------

    def _build_graph(self):
        pipeline = self

        def pre_processing(state: PipelineState) -> PipelineState:
            pipeline.node_pre_processing()
            return state

        def chunking(state: PipelineState) -> PipelineState:
            pipeline.node_chunking()
            return state

        def paraphrasing(state: PipelineState) -> PipelineState:
            pipeline.node_paraphrasing()
            return state

        def triplet_extraction(state: PipelineState) -> PipelineState:
            pipeline.node_triplet_extraction()
            return state

        def triplet_normalization(state: PipelineState) -> PipelineState:
            pipeline.node_triplet_normalization()
            return state

        def entity_extraction(state: PipelineState) -> PipelineState:
            pipeline.node_entity_extraction()
            return state

        def ioc_detection(state: PipelineState) -> PipelineState:
            pipeline.node_ioc_detection()
            return state

        def type_matching(state: PipelineState) -> PipelineState:
            pipeline.node_type_matching()
            return state

        def internal_entity_resolution(state: PipelineState) -> PipelineState:
            pipeline.node_internal_entity_resolution()
            return state

        def existing_entity_resolution(state: PipelineState) -> PipelineState:
            pipeline.node_existing_entity_resolution()
            return state

        def data_insert(state: PipelineState) -> PipelineState:
            pipeline.node_data_insert()
            return state

        builder = StateGraph(PipelineState)
        builder.add_node("pre_processing", pre_processing)
        builder.add_node("chunking", chunking)
        builder.add_node("paraphrasing", paraphrasing)
        builder.add_node("triplet_extraction", triplet_extraction)
        builder.add_node("triplet_normalization", triplet_normalization)
        builder.add_node("entity_extraction", entity_extraction)
        builder.add_node("ioc_detection", ioc_detection)
        builder.add_node("type_matching", type_matching)
        builder.add_node("internal_entity_resolution", internal_entity_resolution)
        builder.add_node("existing_entity_resolution", existing_entity_resolution)
        builder.add_node("data_insert", data_insert)

        builder.set_entry_point("pre_processing")
        builder.add_edge("pre_processing", "chunking")
        builder.add_edge("chunking", "paraphrasing")
        builder.add_edge("paraphrasing", "triplet_extraction")
        builder.add_edge("triplet_extraction", "triplet_normalization")
        builder.add_edge("triplet_normalization", "entity_extraction")
        builder.add_edge("entity_extraction", "ioc_detection")
        builder.add_edge("ioc_detection", "type_matching")
        builder.add_edge("type_matching", "internal_entity_resolution")
        builder.add_edge("internal_entity_resolution", "existing_entity_resolution")
        builder.add_edge("existing_entity_resolution", "data_insert")
        builder.add_edge("data_insert", END)

        return builder.compile()

    def run(self) -> None:
        self._log(logging.INFO, "[START] OntologyExtractor run at %s", to_iso_now())
        graph = self._build_graph()
        graph.invoke({"error": None})
        self._log(logging.INFO, "[DONE] Outputs at %s", self.run_root.resolve())

    def prepare_plain_text_input(self) -> Path:
        """Treat the input source as already-extracted plain text and seed preprocess output."""
        source_path = Path(self.input_source)
        if not source_path.is_file():
            raise FileNotFoundError(f"Plain text input not found: {self.input_source}")

        suffix = source_path.suffix if source_path.suffix else ".txt"
        prepared = self.preprocess_dir / f"{safe_filename(source_path.stem)}{suffix}"
        raw_text = source_path.read_text(encoding="utf-8", errors="ignore")
        normalized = normalize_plain_text(raw_text)
        prepared.write_text(normalized, encoding="utf-8")
        self.preprocess_file = prepared
        self._log(logging.INFO, "[PREP] Plain text input prepared: %s", prepared)
        return prepared

    def run_from_chunking(self) -> None:
        import time as _time
        self._log(logging.INFO, "[START] OntologyExtractor run from chunking at %s", to_iso_now())
        _total_start = _time.monotonic()
        _timings: list = []

        def _run(label: str, fn):
            _t = _time.monotonic()
            fn()
            _elapsed = _time.monotonic() - _t
            _timings.append((label, _elapsed))
            self._log(logging.INFO, "[TIMING] %-40s %dm %05.2fs", label, int(_elapsed) // 60, _elapsed % 60)

        _run("prepare_plain_text_input",       self.prepare_plain_text_input)
        _run("node_chunking",                  self.node_chunking)
        _run("node_paraphrasing",              self.node_paraphrasing)
        _run("node_triplet_extraction",        self.node_triplet_extraction)
        _run("node_triplet_normalization",     self.node_triplet_normalization)
        _run("node_entity_extraction",         self.node_entity_extraction)
        _run("node_ioc_detection",             self.node_ioc_detection)
        _run("node_type_matching",             self.node_type_matching)
        _run("node_internal_entity_resolution",self.node_internal_entity_resolution)
        _run("node_existing_entity_resolution",self.node_existing_entity_resolution)
        _run("node_data_insert",               self.node_data_insert)

        total = _time.monotonic() - _total_start
        self._log(logging.INFO, "[DONE] Outputs at %s", self.run_root.resolve())
        self._log(logging.INFO, "[TIMING] %-40s %dm %05.2fs  ← total", "TOTAL", int(total) // 60, total % 60)
        print(f"\n  ── Pipeline timing ──────────────────────────────")
        for label, t in _timings:
            mm, ss = divmod(t, 60)
            bar = "█" * min(int(t / 10), 20)
            print(f"  {label:<40} {int(mm):>3}m {ss:05.2f}s  {bar}")
        mm, ss = divmod(total, 60)
        print(f"  {'TOTAL':<40} {int(mm):>3}m {ss:05.2f}s")
        print(f"  ─────────────────────────────────────────────────")

    def run_from_chunking_until_internal_entity_resolution(self) -> None:
        import time as _time
        self._log(logging.INFO, "[START] OntologyExtractor run from chunking until internal entity resolution at %s", to_iso_now())
        _total_start = _time.monotonic()
        _timings: list = []

        def _run(label: str, fn):
            _t = _time.monotonic()
            fn()
            _elapsed = _time.monotonic() - _t
            _timings.append((label, _elapsed))
            self._log(logging.INFO, "[TIMING] %-40s %dm %05.2fs", label, int(_elapsed) // 60, _elapsed % 60)

        _run("prepare_plain_text_input", self.prepare_plain_text_input)
        _run("node_chunking",            self.node_chunking)
        _run("node_paraphrasing",        self.node_paraphrasing)
        _run("node_triplet_extraction",  self.node_triplet_extraction)
        _run("node_triplet_normalization", self.node_triplet_normalization)
        _run("node_entity_extraction",   self.node_entity_extraction)
        _run("node_ioc_detection",       self.node_ioc_detection)
        _run("node_type_matching",       self.node_type_matching)
        _run("node_internal_entity_resolution", self.node_internal_entity_resolution)

        total = _time.monotonic() - _total_start
        self._log(logging.INFO, "[DONE] Outputs at %s", self.run_root.resolve())
        self._log(logging.INFO, "[TIMING] %-40s %dm %05.2fs  ← total", "TOTAL", int(total) // 60, total % 60)
        # Also print a compact summary to stdout for visibility
        print(f"\n  ── Pipeline timing ──────────────────────────────")
        for label, t in _timings:
            mm, ss = divmod(t, 60)
            bar = "█" * min(int(t / 10), 20)
            print(f"  {label:<40} {int(mm):>3}m {ss:05.2f}s  {bar}")
        mm, ss = divmod(total, 60)
        print(f"  {'TOTAL':<40} {int(mm):>3}m {ss:05.2f}s")
        print(f"  ─────────────────────────────────────────────────")
