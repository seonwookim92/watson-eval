from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb  # type: ignore

from .clients import EmbeddingClient


class ChromaVectorStore:
    """ChromaDB (SQLite-backed) replacement for the zvec vector store."""

    def __init__(
        self,
        store_path: str,
        zvec_config: Dict[str, Any],
        logger: logging.Logger,
        embedding_client: EmbeddingClient,
    ) -> None:
        self.logger = logger
        self.embedding_client = embedding_client
        self.collection_name = str(zvec_config.get("collection_name", "entities"))
        store_root = Path(store_path)
        store_root.mkdir(parents=True, exist_ok=True)
        self.logger.info("[ChromaDB] Initializing persistent store at %s", store_root)
        self._client = chromadb.PersistentClient(path=str(store_root.resolve()))
        self._col = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.logger.info("[ChromaDB] Collection '%s' ready", self.collection_name)

    def _encode(self, text: str) -> List[float]:
        return self.embedding_client.encode(text)

    def query(self, vector: List[float], entity_type: str, topk: int = 10) -> List[Dict[str, Any]]:
        try:
            results = self._col.query(
                query_embeddings=[vector],
                n_results=max(1, int(topk)),
                where={"entity_type": entity_type},
                include=["metadatas", "distances"],
            )
        except Exception as e:
            self.logger.error("[ChromaDB] Query failed: %s", e)
            return []

        rows: List[Dict[str, Any]] = []
        metadatas = results.get("metadatas", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []
        for meta, dist in zip(metadatas, distances):
            if meta is None:
                continue
            rows.append({
                "name": meta.get("name", ""),
                "entity_type": meta.get("entity_type", ""),
                "score": 1.0 - float(dist),  # cosine: distance→similarity
            })
        return rows

    def upsert(self, name: str, entity_type: str, embedding: List[float]) -> None:
        doc_id = f"{entity_type}::{name}"
        self._col.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[{"name": name, "entity_type": entity_type}],
        )


class VectorStore:
    def __init__(
        self,
        store_path: str,
        zvec_config: Dict[str, Any],
        logger: logging.Logger,
        fallback_embedding: EmbeddingClient,
    ) -> None:
        self.logger = logger
        self.fallback_embedding = fallback_embedding
        self.logger.info("[VectorStore] Initializing")
        self.backend = ChromaVectorStore(store_path, zvec_config, logger, fallback_embedding)
        self.logger.info("[VectorStore] Using ChromaDB backend")

    def encode(self, text: str) -> List[float]:
        try:
            return self.backend._encode(text)
        except Exception:
            return self.fallback_embedding.encode(text)

    def query(self, vector: List[float], entity_type: str, topk: int = 10) -> List[Dict[str, Any]]:
        return self.backend.query(vector, entity_type, topk)

    def upsert(self, name: str, entity_type: str, embedding: List[float]) -> None:
        self.backend.upsert(name, entity_type, embedding)


class Neo4jInserter:
    def __init__(self, config: Dict[str, str], logger: logging.Logger) -> None:
        self.logger = logger
        self.config = config
        self.driver = None
        try:
            from neo4j import GraphDatabase  # type: ignore

            self.driver = GraphDatabase.driver(
                self.config["uri"],
                auth=(self.config["user"], self.config["password"]),
            )
            with self.driver.session() as session:
                session.run("RETURN 1")
            self.logger.info("[Neo4j] Connected")
        except Exception as e:
            self.driver = None
            self.logger.error("[Neo4j] Disabled - %s", e)

    def available(self) -> bool:
        return self.driver is not None

    def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        retries: int = 3,
    ) -> bool:
        if not self.driver:
            return False
        params = params or {}
        for attempt in range(1, retries + 1):
            try:
                with self.driver.session() as session:
                    session.run(query, **params)
                return True
            except Exception as e:
                self.logger.error("[Neo4j] Query failed (attempt %s/%s): %s", attempt, retries, e)
                if attempt == retries:
                    return False
        return False
