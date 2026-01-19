from __future__ import annotations

from typing import Any

from openai import OpenAI

from .config import API_KEY, BASE_URL, MODEL_NAME


class LLMClient:
    def __init__(self) -> None:
        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.model = MODEL_NAME

    def _resolve_model(self) -> str:
        try:
            models = self.client.models.list()
        except Exception:  # pragma: no cover - network/runtime dependent
            return self.model
        available = [item.id for item in getattr(models, "data", []) if getattr(item, "id", None)]
        if not available:
            return self.model
        if self.model in available:
            return self.model
        return available[0]

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> Any:
        model = self._resolve_model()
        self.model = model
        try:
            if tool_choice == "none":
                return self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
            return self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
        except Exception:  # pragma: no cover - network/runtime dependent
            fallback_model = self._resolve_model()
            self.model = fallback_model
            if tool_choice == "none":
                return self.client.chat.completions.create(
                    model=fallback_model,
                    messages=messages,
                )
            return self.client.chat.completions.create(
                model=fallback_model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
