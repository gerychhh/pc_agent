from __future__ import annotations

from typing import Any

from openai import OpenAI

from .config import API_KEY, BASE_URL


class LLMClient:
    def __init__(self) -> None:
        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.model: str | None = None

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model_name: str,
        tool_choice: str = "auto",
    ) -> Any:
        self.model = model_name
        if tool_choice == "none":
            return self.client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
        return self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
