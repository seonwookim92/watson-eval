"""Shared OpenAI-compatible client used by postprocessors."""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Optional

import requests

DEFAULT_POSTPROCESS_MAX_TOKENS = 5000
DEFAULT_QWEN_TEMPERATURE = 0.7
DEFAULT_QWEN_TOP_P = 0.8
DEFAULT_QWEN_TOP_K = 20
DEFAULT_QWEN_PRESENCE_PENALTY = 1.5
DEFAULT_QWEN_REPETITION_PENALTY = 1.0


class LLMClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: Optional[str] = None,
        timeout: int = 120,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)
        self._openai_client = self._load_openai_client()

    def _load_openai_client(self):
        try:
            from openai import OpenAI  # type: ignore

            return OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception:
            return None

    def _post_request(self, payload: dict[str, Any]) -> str:
        payload = {
            **payload,
            "temperature": payload.get("temperature", DEFAULT_QWEN_TEMPERATURE),
            "top_p": payload.get("top_p", DEFAULT_QWEN_TOP_P),
            "top_k": payload.get("top_k", DEFAULT_QWEN_TOP_K),
            "presence_penalty": payload.get(
                "presence_penalty", DEFAULT_QWEN_PRESENCE_PENALTY
            ),
            "repetition_penalty": payload.get(
                "repetition_penalty", DEFAULT_QWEN_REPETITION_PENALTY
            ),
            "chat_template_kwargs": {
                **payload.get("chat_template_kwargs", {}),
                "enable_thinking": False,
            },
        }
        if self._openai_client is not None:
            sdk_payload = dict(payload)
            extra_body = {
                "top_k": sdk_payload.pop("top_k", DEFAULT_QWEN_TOP_K),
                "repetition_penalty": sdk_payload.pop(
                    "repetition_penalty", DEFAULT_QWEN_REPETITION_PENALTY
                ),
                "chat_template_kwargs": sdk_payload.pop("chat_template_kwargs", {}),
            }
            response = self._openai_client.chat.completions.create(
                model=self.model,
                extra_body=extra_body,
                **sdk_payload,
            )
            choices = getattr(response, "choices", [])
            if not choices:
                raise RuntimeError("LLM 응답에서 choices가 비었습니다.")
            message = getattr(choices[0].message, "content", None)
            if not isinstance(message, str) or not message.strip():
                raise RuntimeError("LLM 응답이 비었습니다.")
            return message.strip()

        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, **payload}
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            raise RuntimeError("LLM API 응답이 비었습니다.")
        message = choices[0].get("message", {})
        content = message.get("content", "") if isinstance(message, dict) else ""
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM 응답 본문이 비었습니다.")
        return content.strip()

    def describe_table(self, table_markdown: str) -> str:
        prompt = (
            "Without summarization, describe the markdown table below in detail. For example, if the input is like below\n\n"
            "| IP Address      | Sample Group | Country        | Protocol          | Live in June 2014? |\n"
            "| --------------- | ------------ | -------------- | ----------------- | ------------------ |\n"
            "| 178.21.172.157  | 3            | Greece         | FTP, HTTP         | Yes                |\n"
            "| 188.116.32.164  | 3            | Poland         | FTP, HTTP         | Yes                |\n\n"
            "The result should be like\n"
            "Description : {\"IP Address\" : \"178.21.172.157\", \"Sample Group\":\"3\", \"Country\":\"Greece\", \"Protocol\":\"FTP, HTTP\", \"Live in June 2014?\":\"Yes\"},\n"
            "{\"IP Address\" : \"188.116.32.164\", \"Sample Group\":\"3\", \"Country\":\"Poland\", \"Protocol\":\"FTP, HTTP\", \"Live in June 2014?\":\"Yes\"\n\n"
            "Input : \n"
            f"{table_markdown}"
            "\n\nDescription : "
        )
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": DEFAULT_POSTPROCESS_MAX_TOKENS,
        }
        return self._post_request(payload)

    def describe_image_from_bytes(self, image_bytes: bytes, image_mime: str, prompt: str) -> str:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime};base64,{b64}"},
            },
        ]

        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": DEFAULT_POSTPROCESS_MAX_TOKENS,
        }
        return self._post_request(payload)

    def describe_image_from_url(self, image_url: str, prompt: str) -> str:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        payload = {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": DEFAULT_POSTPROCESS_MAX_TOKENS,
        }
        return self._post_request(payload)
