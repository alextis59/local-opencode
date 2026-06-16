from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)


@dataclass(frozen=True)
class ParsedToolCalls:
    content: str | None
    tool_calls: list[dict[str, Any]]


@dataclass(frozen=True)
class ParsedReasoning:
    content: str | None
    reasoning_content: str | None


def parse_reasoning(text: str | None) -> ParsedReasoning:
    if not text:
        return ParsedReasoning(content=text, reasoning_content=None)

    reasoning_parts = [match.group(1).strip() for match in THINK_RE.finditer(text)]
    reasoning_parts = [part for part in reasoning_parts if part]

    content = THINK_RE.sub("", text)
    open_think_index = content.rfind("<think>")
    if open_think_index != -1:
        reasoning = content[open_think_index + len("<think>") :].strip()
        if reasoning:
            reasoning_parts.append(reasoning)
        content = content[:open_think_index]

    if not reasoning_parts:
        return ParsedReasoning(content=text, reasoning_content=None)

    content = content.strip() or None
    return ParsedReasoning(content=content, reasoning_content="\n\n".join(reasoning_parts))


def parse_tool_calls(text: str | None) -> ParsedToolCalls:
    if not text:
        return ParsedToolCalls(content=text, tool_calls=[])

    tool_calls: list[dict[str, Any]] = []

    for match in TOOL_CALL_RE.finditer(text):
        raw = match.group(1)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue

        name = payload.get("name")
        arguments = payload.get("arguments", {})
        if not isinstance(name, str) or not name:
            continue

        if isinstance(arguments, str):
            argument_text = arguments
        else:
            argument_text = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))

        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": argument_text,
                },
            }
        )

    if not tool_calls:
        return ParsedToolCalls(content=text, tool_calls=[])

    content = TOOL_CALL_RE.sub("", text).strip() or None
    return ParsedToolCalls(content=content, tool_calls=tool_calls)
