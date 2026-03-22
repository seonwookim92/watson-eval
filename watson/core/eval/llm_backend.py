from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping


@dataclass
class ChatResponse:
    content: str


def _message_role(message: Any) -> str:
    if isinstance(message, Mapping):
        return str(message.get("role", "user"))
    message_type = str(getattr(message, "type", "") or "").strip().lower()
    if message_type == "system":
        return "system"
    if message_type in {"ai", "assistant"}:
        return "assistant"
    return "user"


def _message_content(message: Any) -> str:
    if isinstance(message, Mapping):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def _to_openai_messages(messages: Iterable[Any]) -> List[dict[str, str]]:
    return [
        {
            "role": _message_role(message),
            "content": _message_content(message),
        }
        for message in messages
    ]


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


class OpenAIJudgeClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 120,
    ) -> None:
        from openai import AsyncOpenAI

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
        }
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)
        self._model = model

    async def ainvoke(self, messages: Iterable[Any]) -> ChatResponse:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=_to_openai_messages(messages),
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response.choices[0].message.content or ""
        return ChatResponse(content=content)


class LangChainJudgeClient:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def ainvoke(self, messages: Iterable[Any]) -> ChatResponse:
        from langchain_core.messages import HumanMessage, SystemMessage

        converted = []
        for message in messages:
            role = _message_role(message)
            content = _message_content(message)
            if role == "system":
                converted.append(SystemMessage(content=content))
            else:
                converted.append(HumanMessage(content=content))
        response = await self._inner.ainvoke(converted)
        return ChatResponse(content=_flatten_content(getattr(response, "content", "")))


def build_llm_judge(provider: str, model: str, base_url: str | None = None):
    normalized = (provider or "openai").strip().lower()
    if normalized == "openai":
        return OpenAIJudgeClient(
            model=model,
            api_key=os.getenv("OPENAI_API_KEY", "dummy"),
            base_url=base_url,
        )
    if normalized == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return LangChainJudgeClient(
            ChatGoogleGenerativeAI(
                model=model,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0,
            )
        )
    if normalized in ("claude", "anthropic"):
        from langchain_anthropic import ChatAnthropic

        return LangChainJudgeClient(
            ChatAnthropic(
                model=model,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                temperature=0,
            )
        )

    from langchain_ollama import ChatOllama

    return LangChainJudgeClient(
        ChatOllama(
            model=model,
            base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )
    )
