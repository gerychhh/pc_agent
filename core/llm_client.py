from __future__ import annotations

from typing import Any

from openai import OpenAI

from .config import API_KEY, BASE_URL, MODEL_NAME


class LLMClient:
    def __init__(self) -> None:
        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.model = MODEL_NAME

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> Any:
        if tool_choice == "none":
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
