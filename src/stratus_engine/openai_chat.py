from __future__ import annotations

from dataclasses import dataclass

from stratus_engine.metrics import estimate_tokens
from stratus_engine.models import ContextBundle


@dataclass(frozen=True)
class ChatResult:
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIConversationClient:
    def __init__(self, *, model: str = "gpt-4o-mini") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAIConversationClient requires `pip install -e \".[openai]\"`."
            ) from exc

        self.client = OpenAI()
        self.model = model

    def respond(self, *, prompt: str, context: ContextBundle) -> ChatResult:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a practical analyst using Stratus Engine context. "
                    "Explain things clearly, compare approaches when asked, and "
                    "ground your answer in the provided memories and metrics. "
                    "Do not invent numbers or claims."
                ),
            },
            {"role": "system", "content": _render_context(context)},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        answer = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or estimate_tokens(str(messages))
        completion_tokens = getattr(usage, "completion_tokens", 0) or estimate_tokens(answer)
        total_tokens = getattr(usage, "total_tokens", 0) or prompt_tokens + completion_tokens
        return ChatResult(
            answer=answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


def _render_context(context: ContextBundle) -> str:
    sections = context.as_prompt_sections()
    memories = "\n".join(sections["memories"]) or "(none)"
    recent_messages = "\n".join(sections["recent_messages"]) or "(none)"
    return (
        f"Session summary:\n{sections['session_summary'] or '(empty)'}\n\n"
        f"Relevant memories:\n{memories}\n\n"
        f"Recent session messages:\n{recent_messages}"
    )

