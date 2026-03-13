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

import requests
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from .clients import EmbeddingClient, LLMClient, MCPStdioClient
from .config import load_config
from .prompts import (
    class_resolution_agent_prompt,
    chunk_prompt,
    data_property_check_prompt,
    data_property_resolution_agent_prompt,
    entity_extraction_prompt,
    entity_inventory_from_triplets_prompt,
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
    predicate_match_select_prompt,
    triplet_prompt,
    type_match_select_prompt,
)
from .storage import Neo4jInserter, VectorStore
from .utils import (
    clean_text,
    is_url,
    normalize_plain_text,
    safe_filename,
    sanitize_neo4j_identifier,
    setup_logger,
    to_iso_now,
)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    error: Optional[str]


class OntologyConstraintChecker:
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
            if schema_path.is_dir():
                files: List[Tuple[str, str]] = []
                for pattern, fmt in (("**/*.ttl", "ttl"), ("**/*.owl", "xml"), ("**/*.rdf", "xml")):
                    for file_path in glob(str(schema_path / pattern), recursive=True):
                        files.append((file_path, fmt))
                if not files:
                    raise FileNotFoundError(f"No ontology files found under schema directory: {schema_path}")
                for file_path, fmt in files:
                    graph.parse(file_path, format=fmt)
            else:
                graph.parse(str(schema_path))
            self._rdf = {"RDF": RDF, "RDFS": RDFS, "OWL": OWL, "URIRef": URIRef}

            for cls in graph.subjects(RDF.type, OWL.Class):
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

        self.llm = LLMClient(
            self.config["llm"]["base_url"],
            self.config["llm"]["model"],
            max_tokens=int(self.config["llm"].get("max_tokens", 10000)),
            logger=self.logger,
        )
        self.embedding = EmbeddingClient(
            self.config["embedding"]["base_url"],
            self.config["embedding"]["model"],
            self.config["embedding"]["truncate_prompt_tokens"],
            api_key=self.config["embedding"].get("api_key", ""),
        )
        self.neo4j = Neo4jInserter(self.config["neo4j"], self.logger)
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
        self.triplets: List[Dict[str, Any]] = []
        self.all_typed_triplets: List[Dict[str, Any]] = []
        self.typed_triplets: List[Dict[str, Any]] = []
        self.mcp_call_count: Dict[str, int] = {"type_matching": 0, "property_matching": 0}
        self.report_root: Dict[str, str] = {"name": "", "class_uri": "", "class_name": ""}

        # Node 7 caches: avoid redundant MCP calls for identical entities/predicates
        self._entity_class_cache: Dict[str, Tuple[str, str]] = {}
        self._ioc_type_class_cache: Dict[str, Tuple[str, str]] = {}
        self._literal_check_cache: Dict[Tuple[str, str], Tuple[bool, str]] = {}
        self._predicate_uri_cache: Dict[Tuple[str, str, str], str] = {}

        # Locks for thread-safe cache access and shared counter updates
        self._cache_lock = threading.Lock()
        self._mcp_count_lock = threading.Lock()
        self._progress_lock = threading.Lock()
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

    def _read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")

    def _split_sentences(self, text: str) -> List[str]:
        try:
            import nltk
            return [s for s in nltk.sent_tokenize(text) if s.strip()]
        except Exception:
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
        latest_safe = ""
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
            if technical_ok:
                latest_safe = current
            if technical_ok and clear_spo and one_relation:
                return current
        if latest_safe:
            self._log(logging.WARNING, "[3-4] Chunk %s accepted latest technically safe rewrite", idx)
            return latest_safe
        self._log(logging.WARNING, "[3-4] Chunk %s kept input after retries", idx)
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
        # current = self._run_stage4(idx, current, retry_limit)
        # Temporarily bypass stage 3/4 and feed stage 1/2 output directly into extraction.
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
        self._write_json(self.intermediate / "entities.json", {"entities": self.entities})
        self._write_json(self.intermediate / "triplets.json", self.triplets)
        self._log(logging.INFO, "[4] Entity Extraction completed: %s entities", len(self.entities))

    # ------------------------------------------------------------------
    # Node 5: Triplet Extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_entity_key(name: str) -> str:
        return re.sub(r"\s+", " ", clean_text(name).casefold())

    def _sentence_entities(self, chunk_name: str, sentence: str) -> List[Dict[str, str]]:
        sentence_text = f" {re.sub(r'\s+', ' ', sentence.casefold())} "
        matched: List[Dict[str, str]] = []
        for entity in self.entities_by_chunk.get(chunk_name, []):
            name = entity["name"]
            normalized_name = self._normalize_entity_key(name)
            if not normalized_name:
                continue
            pattern = rf"(?<!\w){re.escape(normalized_name)}(?!\w)"
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
            from iocsearcher.searcher import Searcher  # type: ignore

            searcher = Searcher()
            raw_hits = searcher.search_raw(value) or []
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
        except Exception:
            pass
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
        subject_context = row["source_sentence"]
        if row.get("subject_description"):
            subject_context = f'{subject_context}\nEntity description: {row["subject_description"]}'

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
        object_context = row["source_sentence"]
        if row.get("object_description"):
            object_context = f'{object_context}\nEntity description: {row["object_description"]}'

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
            pred_uri = self._match_predicate_uri(
                row["subject"], row["id"], subj_class_uri,
                row["predicate"], row["object"], obj_class_uri,
            )
        else:
            literal_flag, candidate_pred_uri = self._check_is_literal(
                row["subject"], row["id"], subj_class_uri, row["predicate"], row["object"]
            )
            if literal_flag:
                object_is_literal = True
                pred_uri = candidate_pred_uri
            else:
                obj_class_uri, obj_class_name = self._match_entity_class(
                    mcp, row["object"], object_context
                )
                pred_uri = self._match_predicate_uri(
                    row["subject"], row["id"], subj_class_uri,
                    row["predicate"], row["object"], obj_class_uri,
                )

        typed_row = {
            "id": row["id"],
            "source_sentence": row["source_sentence"],
            "source_chunk": row["source_chunk"],
            "subject": row["subject"],
            "subject_description": row.get("subject_description", ""),
            "subject_class_uri": subj_class_uri,
            "subject_class_name": subj_class_name,
            "predicate": row["predicate"],
            "predicate_uri": pred_uri,
            "object": row["object"],
            "object_description": row.get("object_description", ""),
            "object_is_literal": object_is_literal,
            "object_class_uri": obj_class_uri,
            "object_class_name": obj_class_name,
            "isSubjectIoC": row["isSubjectIoC"],
            "isObjectIoC": row["isObjectIoC"],
            "subjectIoCType": row.get("subjectIoCType", ""),
            "objectIoCType": row.get("objectIoCType", ""),
        }
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
                embedding_base_url=str(self.config["embedding"].get("base_url", "")),
                embedding_model=str(self.config["embedding"].get("model", "")),
                embedding_api_key=str(self.config["embedding"].get("api_key", "")),
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

    def _match_entity_class(
        self, mcp: MCPStdioClient, entity: str, context: str
    ) -> Tuple[str, str]:
        cache_key = entity.strip().lower()
        with self._cache_lock:
            if cache_key in self._entity_class_cache:
                return self._entity_class_cache[cache_key]

        max_per_entity = int(self.config["mcp"]["max_tool_calls_type_matching"])
        allowed_tools = {
            "search_classes",
            "list_root_classes",
            "list_subclasses",
            "get_class_hierarchy",
            "get_class_details",
            "list_available_facets",
        }
        validation_feedback = ""
        class_uri = ""
        class_name = ""
        for _ in range(3):
            parsed = self._run_mcp_agent_loop(
                mcp=mcp,
                prompt_factory=lambda transcript, remaining, force_finish=False, feedback=validation_feedback: class_resolution_agent_prompt(
                    entity=entity,
                    context=context,
                    transcript=self._augment_transcript(transcript, feedback),
                    remaining_calls=remaining,
                    force_finish=force_finish,
                ),
                allowed_tools=allowed_tools,
                max_calls=max_per_entity,
                counter_key="type_matching",
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

    def _match_predicate_uri(
        self,
        subject: str,
        row_id: Any,
        subject_class_uri: str,
        predicate: str,
        obj: str,
        obj_class_uri: str,
    ) -> str:
        cache_key = (subject_class_uri, predicate, obj_class_uri)
        with self._cache_lock:
            if cache_key in self._predicate_uri_cache:
                return self._predicate_uri_cache[cache_key]

        max_calls = int(self.config["mcp"]["max_tool_calls_property_matching"])
        validation_feedback = ""
        result = ""
        for _ in range(3):
            parsed = self._run_property_agent_with_temp_entities(
                subject_class_uri=subject_class_uri,
                object_class_uri=obj_class_uri,
                prompt_factory=lambda transcript, remaining, temp_subject_uri, temp_object_uri, force_finish=False, feedback=validation_feedback: object_property_resolution_agent_prompt(
                    subject=subject,
                    subject_class_uri=subject_class_uri,
                    subject_entity_uri=temp_subject_uri,
                    predicate=predicate,
                    obj=obj,
                    obj_class_uri=obj_class_uri,
                    object_entity_uri=temp_object_uri,
                    transcript=self._augment_transcript(transcript, feedback),
                    remaining_calls=remaining,
                    force_finish=force_finish,
                ),
                allowed_tools={"recommend_relation", "search_properties", "get_class_details", "list_available_facets"},
                max_calls=max_calls,
                counter_key="property_matching",
                subject_entity_id=self._temp_mcp_entity_id(row_id, "subject"),
                object_entity_id=self._temp_mcp_entity_id(row_id, "object"),
            )
            result = parsed.get("property_uri", "") if isinstance(parsed, dict) else ""
            if result and not self._response_uri_seen(result, parsed.get("_transcript", "")):
                result = ""
            is_valid, reason = self._validate_property_uri(result, expect_data_property=False if result else None)
            if is_valid:
                break
            validation_feedback = (
                f"{reason} The previous answer is invalid. "
                "Return an object property URI that actually exists in the loaded ontology, or return an empty property URI if no valid relation fits."
            )
            self._log(logging.WARNING, "[MCP-Agent][property_matching] %s", reason)
            result = ""
        with self._cache_lock:
            self._predicate_uri_cache[cache_key] = result
        return result

    def _run_mcp_agent_loop(
        self,
        mcp: MCPStdioClient,
        prompt_factory: Any,
        allowed_tools: set,
        max_calls: int,
        counter_key: str,
    ) -> Dict[str, Any]:
        transcript_entries: List[Dict[str, Any]] = []
        for call_index in range(max_calls):
            remaining = max_calls - call_index
            prompt = prompt_factory(self._format_agent_transcript(transcript_entries), remaining, False)
            self._log(
                logging.INFO,
                "[MCP-Agent][%s][step %s/%s] Prompt:\n%s",
                counter_key,
                call_index + 1,
                max_calls,
                self._truncate_for_log(prompt),
            )
            parsed = self.llm.chat_json(prompt)
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
                return parsed
            if action != "call_tool":
                break

            tool = str(parsed.get("tool", "")).strip()
            if tool not in allowed_tools:
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
        self._log(
            logging.INFO,
            "[MCP-Agent][%s][final] Prompt:\n%s",
            counter_key,
            self._truncate_for_log(final_prompt),
        )
        parsed = self.llm.chat_json(final_prompt)
        self._log(
            logging.INFO,
            "[MCP-Agent][%s][final] LLM response: %s",
            counter_key,
            self._truncate_for_log(json.dumps(parsed, ensure_ascii=False)),
        )
        if not isinstance(parsed, dict):
            parsed = {}
        parsed["_transcript"] = self._format_agent_transcript(transcript_entries)
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
    ) -> str:
        with self._mcp_count_lock:
            self.mcp_call_count[counter_key] += 1
        self._log(
            logging.INFO,
            "[MCP-Agent][%s][setup] Tool call: %s %s",
            counter_key,
            tool,
            self._truncate_for_log(json.dumps(arguments, ensure_ascii=False)),
        )
        result = mcp.call_tool(tool, arguments)
        self._log(
            logging.INFO,
            "[MCP-Agent][%s][setup] Tool result from %s:\n%s",
            counter_key,
            tool,
            self._truncate_for_log(result),
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
        try:
            with mcp:
                self._call_mcp_tool(
                    mcp,
                    "create_entity",
                    {"entity_id": subject_entity_id, "class_uris": [subject_class_uri]},
                    counter_key,
                )
                if object_entity_id and object_class_uri:
                    self._call_mcp_tool(
                        mcp,
                        "create_entity",
                        {"entity_id": object_entity_id, "class_uris": [object_class_uri]},
                        counter_key,
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

        for name, class_uri in unique:
            try:
                vec = self.vector_store.encode(name)
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
    # Node 10: Data Insert
    # ------------------------------------------------------------------

    def node_data_insert(self) -> None:
        self._log(logging.INFO, "[10] Data Insert start")
        if not self.neo4j.available():
            self._log(logging.ERROR, "[10] Neo4j unavailable; skip insert")
            return

        retry = int(self.config["retry"]["data_insert"])

        # 9-1: Insert report root node
        if self.report_root.get("class_uri") or self.report_root.get("class_name"):
            self._log(logging.INFO, "[10-1] Insert report root node: %s", self.source_filename)
            root_label = sanitize_neo4j_identifier(
                self.report_root["class_name"] or self.report_root["class_uri"].rsplit("#", 1)[-1],
                fallback="Report",
            )
            self.neo4j.execute(
                f"MERGE (r:OntologyEntity:`{root_label}` {{name: $name, class_uri: $class_uri}})\n"
                "SET r.source_document = $source_document, "
                "r.entity_type = $entity_type, r.class_name = $class_name, r.is_report_root = true",
                {
                    "name": self.source_filename,
                    "source_document": self.source_filename,
                    "class_uri": self.report_root["class_uri"],
                    "entity_type": self.report_root["class_uri"],
                    "class_name": self.report_root["class_name"],
                },
                retries=retry,
            )

        # Collect unique entity nodes (exclude literal objects)
        nodes: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for trip in self.typed_triplets:
            key_s = (trip["subject"], trip["subject_class_uri"])
            if key_s not in nodes:
                nodes[key_s] = {
                    "name": trip["subject"],
                    "class_uri": trip["subject_class_uri"],
                    "class_name": trip["subject_class_name"],
                }
            if not trip["object_is_literal"]:
                key_o = (trip["object"], trip["object_class_uri"])
                if key_o not in nodes:
                    nodes[key_o] = {
                        "name": trip["object"],
                        "class_uri": trip["object_class_uri"],
                        "class_name": trip["object_class_name"],
                    }

        batch = int(self.config.get("parallel_batch_size", 6))

        # 9-2: Insert nodes in parallel
        typed_nodes = [n for n in nodes.values() if n["class_uri"]]
        total_nodes = len(typed_nodes)
        success_nodes = 0
        with ThreadPoolExecutor(max_workers=batch) as executor:
            futures = [executor.submit(self._insert_node, node, retry) for node in typed_nodes]
            for future in futures:
                try:
                    if future.result():
                        success_nodes += 1
                except Exception as e:
                    self._log(logging.ERROR, "[10-2] Node insert exception: %s", e)

        # 9-3: Insert edges in parallel
        total_edges = len(self.typed_triplets)
        success_edges = 0
        with ThreadPoolExecutor(max_workers=batch) as executor:
            futures = [executor.submit(self._insert_edge, trip, retry) for trip in self.typed_triplets]
            for future in futures:
                try:
                    if future.result():
                        success_edges += 1
                except Exception as e:
                    self._log(logging.ERROR, "[10-3] Edge insert exception: %s", e)

        report_links_total = 0
        report_links_success = 0
        report_mentions_fallback_total = 0
        report_mentions_fallback_success = 0
        if self.report_root.get("class_uri"):
            with ThreadPoolExecutor(max_workers=batch) as executor:
                futures = {}
                for node in typed_nodes:
                    if (
                        node["name"] == self.report_root.get("name")
                        and node["class_uri"] == self.report_root.get("class_uri")
                    ):
                        continue
                    selected = self._select_report_link(node["class_uri"])
                    if not selected:
                        report_mentions_fallback_total += 1
                        futures[
                            executor.submit(
                                self._insert_report_mentions_fallback,
                                node,
                                retry,
                            )
                        ] = "fallback"
                        continue
                    property_uri, report_to_entity = selected
                    report_links_total += 1
                    futures[
                        executor.submit(
                            self._insert_report_link,
                            node,
                            property_uri,
                            report_to_entity,
                            retry,
                        )
                    ] = "schema"
                for future, link_type in futures.items():
                    try:
                        if future.result():
                            if link_type == "schema":
                                report_links_success += 1
                            else:
                                report_mentions_fallback_success += 1
                    except Exception as e:
                        self._log(logging.ERROR, "[10-3b] Report link insert exception: %s", e)

        # 9-4: Save embeddings in parallel
        def _save_embedding(name: str, entity_type: str) -> bool:
            try:
                vec = self.vector_store.encode(name)
                self.vector_store.upsert(name, entity_type, vec)
                return True
            except Exception as e:
                self._log(logging.WARNING, "[10-4] Embedding save failed for %s: %s", name, e)
                return False

        saved_embeddings = 0
        embed_targets = [(name, et) for (name, et) in nodes.keys() if et]
        with ThreadPoolExecutor(max_workers=batch) as executor:
            futures = [executor.submit(_save_embedding, name, et) for name, et in embed_targets]
            for future in futures:
                try:
                    if future.result():
                        saved_embeddings += 1
                except Exception as e:
                    self._log(logging.ERROR, "[10-4] Embedding exception: %s", e)

        self._log(logging.INFO, "[Data Insert] Nodes: %s/%s inserted", success_nodes, total_nodes)
        self._log(logging.INFO, "[Data Insert] Edges: %s/%s inserted", success_edges, total_edges)
        self._log(logging.INFO, "[Data Insert] Report links: %s/%s inserted", report_links_success, report_links_total)
        self._log(
            logging.INFO,
            "[Data Insert] Report mentions fallback: %s/%s inserted",
            report_mentions_fallback_success,
            report_mentions_fallback_total,
        )
        self._log(logging.INFO, "[Data Insert] Embeddings: %s/%s saved to zvec", saved_embeddings, total_nodes)
        self._log(logging.INFO, "[10] Data Insert complete")

    # ------------------------------------------------------------------
    # Node 10 Cypher helpers — template-based, no LLM involved
    # ------------------------------------------------------------------

    def _insert_node(self, node: Dict[str, Any], retry: int) -> bool:
        name = node["name"]
        class_uri = node["class_uri"]
        class_name = node["class_name"]
        class_label = sanitize_neo4j_identifier(
            class_name or class_uri.rsplit("#", 1)[-1],
            fallback="OntologyEntity",
        )
        cypher = (
            f"MERGE (n:OntologyEntity:`{class_label}` {{name: $name, class_uri: $class_uri}})\n"
            "SET n.entity_type = $entity_type, n.class_name = $class_name, "
            "n.source_document = $source_document"
        )
        return self.neo4j.execute(
            cypher,
            {
                "name": name,
                "class_uri": class_uri,
                "entity_type": class_uri,
                "class_name": class_name,
                "source_document": self.source_filename,
                },
                retries=retry,
            )

    @staticmethod
    def _score_report_link_property(property_uri: str) -> Tuple[int, int, str]:
        local_name = property_uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1].lower()
        preferred_keywords = (
            "describ",
            "mention",
            "refer",
            "about",
            "contain",
            "include",
            "document",
            "report",
            "analysis",
        )
        score = 0
        for index, keyword in enumerate(preferred_keywords):
            if keyword in local_name:
                score += len(preferred_keywords) - index
        return score, -len(local_name), local_name

    def _select_report_link(
        self, entity_class_uri: str
    ) -> Optional[Tuple[str, bool]]:
        report_class_uri = str(self.report_root.get("class_uri", "") or "")
        if not (report_class_uri and entity_class_uri):
            return None

        forward = self.ontology_checker.find_object_properties(report_class_uri, entity_class_uri)
        reverse = self.ontology_checker.find_object_properties(entity_class_uri, report_class_uri)
        candidates: List[Tuple[str, bool]] = [(uri, True) for uri in forward]
        candidates.extend((uri, False) for uri in reverse)
        if not candidates:
            return None

        return max(
            candidates,
            key=lambda item: (
                *self._score_report_link_property(item[0]),
                1 if item[1] else 0,
            ),
        )

    def _insert_report_link(
        self,
        node: Dict[str, Any],
        property_uri: str,
        report_to_entity: bool,
        retry: int,
    ) -> bool:
        report_class_uri = str(self.report_root.get("class_uri", "") or "")
        report_class_name = str(self.report_root.get("class_name", "") or "")
        if not (report_class_uri and node.get("class_uri")):
            return False

        report_label = sanitize_neo4j_identifier(
            report_class_name or report_class_uri.rsplit("#", 1)[-1],
            fallback="Report",
        )
        entity_label = sanitize_neo4j_identifier(
            node["class_name"] or node["class_uri"].rsplit("#", 1)[-1],
            fallback="OntologyEntity",
        )
        predicate_name = property_uri.split("#")[-1] if "#" in property_uri else property_uri.rsplit("/", 1)[-1]
        predicate_label = sanitize_neo4j_identifier(predicate_name, fallback="ONTOLOGY_RELATION")
        source_alias, target_alias = ("r", "e") if report_to_entity else ("e", "r")
        cypher = (
            f"MERGE (r:OntologyEntity:`{report_label}` {{name: $report_name, class_uri: $report_class_uri}})\n"
            f"MERGE (e:OntologyEntity:`{entity_label}` {{name: $entity_name, class_uri: $entity_class_uri}})\n"
            f"MERGE ({source_alias})-[rel:`{predicate_label}` {{predicate_uri: $predicate_uri}}]->({target_alias})\n"
            "SET r.source_document = $source_document, r.entity_type = $report_class_uri, "
            "r.class_name = $report_class_name, r.is_report_root = true\n"
            "SET e.source_document = $source_document, e.entity_type = $entity_class_uri, "
            "e.class_name = $entity_class_name\n"
            "SET rel.predicate = $predicate_name, rel.predicate_name = $predicate_name, "
            "rel.predicate_uri = $predicate_uri, rel.source_document = $source_document, "
            "rel.source_sentence = '[report-root link]', rel.source_chunk = '[schema-aware-root-link]'"
        )
        return self.neo4j.execute(
            cypher,
            {
                "report_name": self.report_root["name"],
                "report_class_uri": report_class_uri,
                "report_class_name": report_class_name,
                "entity_name": node["name"],
                "entity_class_uri": node["class_uri"],
                "entity_class_name": node["class_name"],
                "predicate_uri": property_uri,
                "predicate_name": predicate_name,
                "source_document": self.source_filename,
            },
            retries=retry,
        )

    def _insert_report_mentions_fallback(
        self,
        node: Dict[str, Any],
        retry: int,
    ) -> bool:
        report_class_uri = str(self.report_root.get("class_uri", "") or "")
        report_class_name = str(self.report_root.get("class_name", "") or "")
        if not (report_class_uri and node.get("class_uri")):
            return False

        report_label = sanitize_neo4j_identifier(
            report_class_name or report_class_uri.rsplit("#", 1)[-1],
            fallback="Report",
        )
        entity_label = sanitize_neo4j_identifier(
            node["class_name"] or node["class_uri"].rsplit("#", 1)[-1],
            fallback="OntologyEntity",
        )
        cypher = (
            f"MERGE (r:OntologyEntity:`{report_label}` {{name: $report_name, class_uri: $report_class_uri}})\n"
            f"MERGE (e:OntologyEntity:`{entity_label}` {{name: $entity_name, class_uri: $entity_class_uri}})\n"
            "MERGE (r)-[rel:MENTIONS {predicate_uri: $predicate_uri}]->(e)\n"
            "SET r.source_document = $source_document, r.entity_type = $report_class_uri, "
            "r.class_name = $report_class_name, r.is_report_root = true\n"
            "SET e.source_document = $source_document, e.entity_type = $entity_class_uri, "
            "e.class_name = $entity_class_name\n"
            "SET rel.predicate = 'mentions', rel.predicate_name = 'MENTIONS', "
            "rel.predicate_uri = $predicate_uri, rel.source_document = $source_document, "
            "rel.source_sentence = '[report-root fallback link]', rel.source_chunk = '[report-mentions-fallback]'"
        )
        return self.neo4j.execute(
            cypher,
            {
                "report_name": self.report_root["name"],
                "report_class_uri": report_class_uri,
                "report_class_name": report_class_name,
                "entity_name": node["name"],
                "entity_class_uri": node["class_uri"],
                "entity_class_name": node["class_name"],
                "predicate_uri": "fallback:MENTIONS",
                "source_document": self.source_filename,
            },
            retries=retry,
        )

    def _insert_edge(self, trip: Dict[str, Any], retry: int) -> bool:
        subject_label = sanitize_neo4j_identifier(
            trip["subject_class_name"] or trip["subject_class_uri"].rsplit("#", 1)[-1],
            fallback="OntologyEntity",
        )
        predicate_name = trip["predicate_uri"].split("#")[-1] if trip["predicate_uri"] else trip["predicate"]
        predicate_label = sanitize_neo4j_identifier(predicate_name, fallback="ONTOLOGY_RELATION")
        if not trip["object_is_literal"]:
            object_label = sanitize_neo4j_identifier(
                trip["object_class_name"] or trip["object_class_uri"].rsplit("#", 1)[-1],
                fallback="OntologyEntity",
            )
            cypher = (
                f"MERGE (s:OntologyEntity:`{subject_label}` {{name: $subject_name, class_uri: $subject_class_uri}})\n"
                f"MERGE (o:OntologyEntity:`{object_label}` {{name: $object_name, class_uri: $object_class_uri}})\n"
                f"MERGE (s)-[r:`{predicate_label}` {{predicate_uri: $predicate_uri}}]->(o)\n"
                "SET s.entity_type = $subject_class_uri, s.class_name = $subject_class_name, s.source_document = $source_document, "
                "o.entity_type = $object_class_uri, o.class_name = $object_class_name, o.source_document = $source_document\n"
                "SET r.predicate = $predicate, r.predicate_name = $predicate_name, "
                "r.predicate_uri = $predicate_uri, "
                "r.source_sentence = $source_sentence, r.source_chunk = $source_chunk"
            )
            return self.neo4j.execute(
                cypher,
                {
                    "subject_name": trip["subject"],
                    "subject_class_uri": trip["subject_class_uri"],
                    "subject_class_name": trip["subject_class_name"],
                    "object_name": trip["object"],
                    "object_class_uri": trip["object_class_uri"],
                    "object_class_name": trip["object_class_name"],
                    "predicate": trip["predicate"],
                    "predicate_name": predicate_name,
                    "predicate_uri": trip["predicate_uri"],
                    "source_sentence": trip["source_sentence"],
                    "source_chunk": trip["source_chunk"],
                    "source_document": self.source_filename,
                },
                retries=retry,
            )
        else:
            property_name = (
                trip["predicate_uri"].split("#")[-1]
                if "#" in trip["predicate_uri"]
                else trip["predicate_uri"].rsplit("/", 1)[-1]
            ) or predicate_name
            property_key = sanitize_neo4j_identifier(property_name, fallback="literal_value")
            cypher = (
                f"MERGE (s:OntologyEntity:`{subject_label}` {{name: $subject_name, class_uri: $subject_class_uri}})\n"
                "SET s.entity_type = $subject_class_uri, s.class_name = $subject_class_name, s.source_document = $source_document\n"
                f"SET s.`{property_key}` = coalesce(s.`{property_key}`, []) + "
                f"CASE WHEN $object_value IN coalesce(s.`{property_key}`, []) THEN [] ELSE [$object_value] END"
            )
            return self.neo4j.execute(
                cypher,
                {
                    "subject_name": trip["subject"],
                    "subject_class_uri": trip["subject_class_uri"],
                    "subject_class_name": trip["subject_class_name"],
                    "object_value": trip["object"],
                    "predicate": trip["predicate"],
                    "predicate_name": predicate_name,
                    "predicate_uri": trip["predicate_uri"],
                    "source_sentence": trip["source_sentence"],
                    "source_chunk": trip["source_chunk"],
                    "source_document": self.source_filename,
                },
                retries=retry,
            )

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

        def entity_extraction(state: PipelineState) -> PipelineState:
            pipeline.node_entity_extraction()
            return state

        def triplet_extraction(state: PipelineState) -> PipelineState:
            pipeline.node_triplet_extraction()
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
        builder.add_node("entity_extraction", entity_extraction)
        builder.add_node("triplet_extraction", triplet_extraction)
        builder.add_node("ioc_detection", ioc_detection)
        builder.add_node("type_matching", type_matching)
        builder.add_node("internal_entity_resolution", internal_entity_resolution)
        builder.add_node("existing_entity_resolution", existing_entity_resolution)
        builder.add_node("data_insert", data_insert)

        builder.set_entry_point("pre_processing")
        builder.add_edge("pre_processing", "chunking")
        builder.add_edge("chunking", "paraphrasing")
        builder.add_edge("paraphrasing", "triplet_extraction")
        builder.add_edge("triplet_extraction", "entity_extraction")
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
        self._log(logging.INFO, "[START] OntologyExtractor run from chunking at %s", to_iso_now())
        self.prepare_plain_text_input()
        self.node_chunking()
        self.node_paraphrasing()
        self.node_triplet_extraction()
        self.node_entity_extraction()
        self.node_ioc_detection()
        self.node_type_matching()
        self.node_internal_entity_resolution()
        self.node_existing_entity_resolution()
        self.node_data_insert()
        self._log(logging.INFO, "[DONE] Outputs at %s", self.run_root.resolve())

    def run_from_chunking_until_internal_entity_resolution(self) -> None:
        self._log(logging.INFO, "[START] OntologyExtractor run from chunking until internal entity resolution at %s", to_iso_now())
        self.prepare_plain_text_input()
        self.node_chunking()
        self.node_paraphrasing()
        self.node_triplet_extraction()
        self.node_entity_extraction()
        self.node_ioc_detection()
        self.node_type_matching()
        self.node_internal_entity_resolution()
        self._log(logging.INFO, "[DONE] Outputs at %s", self.run_root.resolve())
