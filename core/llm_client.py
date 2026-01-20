from __future__ import annotations

from typing import Any

from openai import OpenAI

from .config import API_KEY, BASE_URL
from .debug import debug_context, debug_event, info_event


class LLMClient:
    def __init__(self) -> None:
        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.model: str | None = None

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content") or ""
            if role == "assistant":
                normalized.append({"role": "assistant", "content": content})
                continue
            if role == "user":
                normalized.append({"role": "user", "content": content})
                continue
            if role == "system":
                normalized.append({"role": "user", "content": f"[SYSTEM]\n{content}".strip()})
                continue
            normalized.append({"role": "user", "content": f"[{role.upper()}]\n{content}".strip()})
        return normalized

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model_name: str,
        tool_choice: str = "auto",
    ) -> Any:
        model_name = self._resolve_model_name(model_name)
        self.model = model_name
        normalized_messages = self._normalize_messages(messages)
        debug_event("LLM_REQ", f"model={model_name} tool_choice={tool_choice}")
        debug_context("LLM_REQ", normalized_messages, limit=1200)
        info_event("LLM_REQ_FULL", str(normalized_messages))
        if tool_choice == "none":
            response = self.client.chat.completions.create(
                model=model_name,
                messages=normalized_messages,
            )
        else:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=normalized_messages,
                tools=tools,
                tool_choice=tool_choice,
            )
        content = response.choices[0].message.content or ""
        debug_context("LLM_RES", content, limit=1200)
        info_event("LLM_RES_FULL", content)
        return response

    def _resolve_model_name(self, model_name: str) -> str:
        if model_name:
            return model_name
        if self.model:
            return self.model
        try:
            models = self.client.models.list()
            if models.data:
                return models.data[0].id
        except Exception:
            pass
        return "default"
