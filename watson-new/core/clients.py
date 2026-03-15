from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .utils import read_json_payload


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 300,
        max_tokens: int = 2048,
        thinking: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.endpoint = f"{self.base_url}/chat/completions"
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def _truncate_for_log(value: Any, limit: int = 4000) -> str:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if len(text) <= limit:
            return text
        return f"{text[:limit]}... [truncated {len(text) - limit} chars]"

    def _request(self, messages: List[Dict[str, str]], response_format_json: bool = True) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_tokens": self.max_tokens,
        }
        if not self.thinking:
            # vLLM: disable Qwen3 thinking via chat_template_kwargs
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}
        last_exc: Exception = RuntimeError("LLM request failed")
        for attempt in range(1, 4):
            try:
                resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    raise RuntimeError(f"Invalid chat response format: {data}")
            except requests.exceptions.Timeout:
                self.logger.error(
                    "[LLM][timeout] model=%s attempt=%s timeout=%ss",
                    self.model,
                    attempt,
                    self.timeout,
                )
                # Timeout: raise immediately, no point retrying same payload
                raise
            except Exception as e:
                self.logger.warning(
                    "[LLM][error] model=%s attempt=%s error=%s",
                    self.model,
                    attempt,
                    e,
                )
                last_exc = e
                if attempt < 3:
                    time.sleep(2 ** attempt)
        raise last_exc

    def chat_json(
        self,
        prompt: str,
        extra_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        if extra_messages:
            messages = extra_messages + messages
        content = self._request(messages, response_format_json=True)
        return read_json_payload(content)

    def chat_text(
        self,
        prompt: str,
        extra_messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        if extra_messages:
            messages = extra_messages + messages
        return self._request(messages, response_format_json=False)


class EmbeddingClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        truncate_prompt_tokens: int,
        timeout: int = 120,
        api_key: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.endpoint = f"{self.base_url}/embeddings"
        self.truncate_prompt_tokens = truncate_prompt_tokens
        self.timeout = timeout
        self.api_key = api_key

    def encode(self, text: str) -> List[float]:
        payload = {
            "input": text,
            "model": self.model,
            "truncate_prompt_tokens": self.truncate_prompt_tokens,
        }
        resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        vector = data.get("data", [{}])[0].get("embedding")
        if not vector:
            raise RuntimeError("Embedding API returned empty vector")
        return vector

    def encode_many(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        payload = {
            "input": texts,
            "model": self.model,
            "truncate_prompt_tokens": self.truncate_prompt_tokens,
        }
        try:
            resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("data", [])
            vectors: List[List[float]] = []
            for item in items:
                vector = item.get("embedding")
                if not vector:
                    raise RuntimeError("Embedding API returned empty vector in batch response")
                vectors.append(vector)
            if len(vectors) != len(texts):
                raise RuntimeError("Embedding batch response length mismatch")
            return vectors
        except Exception:
            return [self.encode(text) for text in texts]


class MCPStdioClient:
    """MCP JSON-RPC client communicating over stdio with the universal-ontology-mcp subprocess."""

    def __init__(
        self,
        script_path: str,
        ontology_file: str,
        logger: logging.Logger,
        embedding_base_url: str = "",
        embedding_model: str = "",
        embedding_api_key: str = "",
        timeout: int = 120,
    ) -> None:
        self.script_path = os.path.abspath(script_path)
        ontology_path = Path(ontology_file).resolve()
        self.ontology_dir = str(ontology_path if ontology_path.is_dir() else ontology_path.parent)
        self.logger = logger
        self.embedding_base_url = embedding_base_url.rstrip("/")
        self.embedding_model = embedding_model
        self.embedding_api_key = embedding_api_key
        self.timeout = timeout
        self.proc: Optional[subprocess.Popen] = None
        self._id = 1
        # Single subprocess → all send/receive must be serialised across threads
        self._send_lock = threading.Lock()

    def __enter__(self) -> "MCPStdioClient":
        env = {**os.environ, "ONTOLOGY_DIR": self.ontology_dir}
        if self.embedding_base_url:
            env["EMBEDDING_API_URL"] = f"{self.embedding_base_url}/embeddings"
        if self.embedding_model:
            env["EMBEDDING_MODEL"] = self.embedding_model
        if self.embedding_api_key:
            env["EMBEDDING_API_KEY"] = self.embedding_api_key
        self.proc = subprocess.Popen(
            [sys.executable, self.script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=False,
            cwd=str(Path(self.script_path).resolve().parent),
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        self._rpc_initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.proc:
            return
        try:
            self._rpc_notification("notifications/exit", {})
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass
        self.proc = None

    @property
    def available(self) -> bool:
        return self.proc is not None

    def _drain_stderr(self) -> None:
        if not self.proc:
            return
        for raw in iter(self.proc.stderr.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                self.logger.debug("[MCP-stderr] %s", line)

    def _read_message(self) -> Dict[str, Any]:
        """Read one NDJSON message from MCP stdout (mcp library >= 1.x uses newline-delimited JSON)."""
        if not self.proc or not self.proc.stdout:
            raise RuntimeError("MCP process is not running")
        while True:
            raw = self.proc.stdout.readline()
            if not raw:
                raise RuntimeError("MCP process closed stdout unexpectedly")
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue  # skip blank lines
            return json.loads(line)

    def _read_result_for(self, request_id: int) -> Dict[str, Any]:
        while True:
            data = self._read_message()
            if isinstance(data, dict) and data.get("id") == request_id:
                return data

    def _send(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        with_id: bool = True,
    ) -> Dict[str, Any]:
        with self._send_lock:
            if not self.proc or not self.proc.stdin:
                raise RuntimeError("MCP process is not running")
            request_id = self._id if with_id else None
            if with_id:
                self._id += 1
            payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
            if with_id:
                payload["id"] = request_id
            if params is not None:
                payload["params"] = params
            data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
            if not with_id:
                return {}
            response = self._read_result_for(request_id)
            if "error" in response:
                raise RuntimeError(f"MCP call failed: {response['error']}")
            return response.get("result", {})

    def _rpc_notification(self, method: str, params: Dict[str, Any]) -> None:
        self._send(method, params, with_id=False)

    def _rpc_initialize(self) -> None:
        init_payload = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "ontology-extractor", "version": "1.0"},
        }
        result = self._send("initialize", init_payload)
        self.logger.info("[MCP] Initialized protocol version: %s", result.get("protocolVersion", "unknown"))
        self._rpc_notification("notifications/initialized", {})

    def list_tools(self) -> Dict[str, Any]:
        return self._send("tools/list")

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        result = self._send("tools/call", {"name": name, "arguments": arguments})
        content = result.get("content", [])
        if isinstance(content, list):
            texts = [
                str(item["text"]) if isinstance(item, dict) and "text" in item else str(item)
                for item in content
            ]
            return "\n".join(texts).strip()
        if isinstance(content, str):
            return content
        return str(result)
