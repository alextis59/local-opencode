from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .tool_calls import parse_reasoning, parse_tool_calls


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


MODEL_ID = os.getenv("VIBETHINKER_MODEL_ID", "vibethinker-3b")
MODEL_PATH = Path(os.getenv("VIBETHINKER_MODEL_PATH", "models/vibethinker-3b-q4_k_m.gguf"))
DEFAULT_MAX_TOKENS = int(os.getenv("VIBETHINKER_DEFAULT_MAX_TOKENS", "64"))
MAX_TOKENS = int(os.getenv("VIBETHINKER_MAX_TOKENS", "64"))
TEMPERATURE = _env_float("VIBETHINKER_TEMPERATURE", 0.2)
TOP_P = _env_float("VIBETHINKER_TOP_P", 0.95)
TOP_K = _env_int("VIBETHINKER_TOP_K", 40)
MIN_P = _env_float("VIBETHINKER_MIN_P", 0.05)
REPEAT_PENALTY = _env_float("VIBETHINKER_REPEAT_PENALTY", 1.0)
BUFFER_STREAM = (
    os.getenv("VIBETHINKER_BUFFER_STREAM")
    or os.getenv("VIBETHINKER_BUFFER_STREAM_WITH_TOOLS")
    or "true"
).lower() != "false"
FALLBACK_TO_REASONING = os.getenv("VIBETHINKER_FALLBACK_TO_REASONING", "true").lower() != "false"
FORWARD_TOOLS = os.getenv("VIBETHINKER_FORWARD_TOOLS", "false").lower() == "true"
FORWARD_TOOL_NAMES = {
    name.strip()
    for name in os.getenv("VIBETHINKER_FORWARD_TOOL_NAMES", "").split(",")
    if name.strip()
}
RESPONSE_FORMAT = os.getenv("VIBETHINKER_RESPONSE_FORMAT", "").strip()
LOG_COMPLETIONS = os.getenv("VIBETHINKER_LOG_COMPLETIONS", "false").lower() == "true"

app = FastAPI(title="VibeThinker local OpenCode gateway", version="0.1.0")
GENERATION_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def load_llm() -> Any:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run `python scripts/download_model.py` first."
        )

    from llama_cpp import Llama

    kwargs: dict[str, Any] = {
        "model_path": str(MODEL_PATH),
        "n_ctx": _env_int("VIBETHINKER_N_CTX", 8192),
        "n_batch": _env_int("VIBETHINKER_N_BATCH", 512),
        "n_gpu_layers": _env_int("VIBETHINKER_N_GPU_LAYERS", 0),
        "verbose": _env_bool("VIBETHINKER_VERBOSE", False),
    }

    n_threads = _env_int("VIBETHINKER_N_THREADS", 0)
    if n_threads > 0:
        kwargs["n_threads"] = n_threads

    chat_format = os.getenv("VIBETHINKER_CHAT_FORMAT")
    if chat_format:
        kwargs["chat_format"] = chat_format

    return Llama(**kwargs)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "model": MODEL_ID,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
    }


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    payload = await request.json()
    if payload.get("model") not in {None, MODEL_ID}:
        raise HTTPException(status_code=404, detail=f"Unknown model: {payload.get('model')}")

    stream = bool(payload.get("stream", False))
    should_buffer_stream = stream and BUFFER_STREAM

    if stream:
        if should_buffer_stream:
            return StreamingResponse(
                stream_buffered_completion(payload),
                media_type="text/event-stream",
            )

        try:
            completion = create_completion(payload, stream=True)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return StreamingResponse(stream_native_completion(completion), media_type="text/event-stream")

    try:
        completion = create_completion(payload, stream=False)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return JSONResponse(normalize_completion(completion))


def create_completion(payload: dict[str, Any], *, stream: bool) -> Any:
    llm = load_llm()
    kwargs = build_completion_kwargs(payload, stream=stream)

    if LOG_COMPLETIONS:
        forwarded_tools = kwargs.get("tools") or []
        forwarded_tool_names = [
            tool.get("function", {}).get("name")
            for tool in forwarded_tools
            if isinstance(tool, dict)
        ]
        print(
            "request summary "
            f"stream={stream} "
            f"messages={len(payload.get('messages') or [])} "
            f"tools={len(payload.get('tools') or [])} "
            f"forwarded_tools={len(forwarded_tools)} "
            f"forwarded_tool_names={forwarded_tool_names} "
            f"forward_tools={FORWARD_TOOLS} "
            f"response_format={RESPONSE_FORMAT or None} "
            f"max_tokens={kwargs.get('max_tokens')} "
            f"temperature={kwargs.get('temperature')} "
            f"top_p={kwargs.get('top_p')} "
            f"top_k={kwargs.get('top_k')} "
            f"min_p={kwargs.get('min_p')} "
            f"repeat_penalty={kwargs.get('repeat_penalty')}",
            flush=True,
        )

    if stream:
        return llm.create_chat_completion(**kwargs)

    with GENERATION_LOCK:
        return llm.create_chat_completion(**kwargs)


def build_completion_kwargs(payload: dict[str, Any], *, stream: bool) -> dict[str, Any]:
    requested_max_tokens = (
        payload.get("max_completion_tokens") or payload.get("max_tokens") or DEFAULT_MAX_TOKENS
    )
    max_tokens = int(requested_max_tokens)
    if MAX_TOKENS > 0:
        max_tokens = min(max_tokens, MAX_TOKENS)

    kwargs: dict[str, Any] = {
        "messages": payload.get("messages", []),
        "temperature": payload.get("temperature", TEMPERATURE),
        "top_p": payload.get("top_p", TOP_P),
        "top_k": payload.get("top_k", TOP_K),
        "min_p": payload.get("min_p", MIN_P),
        "repeat_penalty": payload.get("repeat_penalty", REPEAT_PENALTY),
        "max_tokens": max_tokens,
        "stream": stream,
    }

    for key in ("stop", "presence_penalty", "frequency_penalty"):
        if key in payload:
            kwargs[key] = payload[key]

    if FORWARD_TOOLS:
        tools = filtered_tools(payload.get("tools"))
        if tools:
            kwargs["tools"] = tools
            if "tool_choice" in payload:
                kwargs["tool_choice"] = payload["tool_choice"]

    if RESPONSE_FORMAT:
        kwargs["response_format"] = {"type": RESPONSE_FORMAT}

    return kwargs


def filtered_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    if not FORWARD_TOOL_NAMES:
        return [tool for tool in tools if isinstance(tool, dict)]

    allowed: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        if function.get("name") in FORWARD_TOOL_NAMES:
            allowed.append(tool)
    return allowed


def normalize_completion(completion: dict[str, Any]) -> dict[str, Any]:
    response = dict(completion)
    response["model"] = MODEL_ID

    for choice in response.get("choices", []):
        message = choice.get("message") or {}
        parsed = parse_tool_calls(message.get("content"))
        reasoning = parse_reasoning(parsed.content)

        if reasoning.reasoning_content:
            message["reasoning_content"] = reasoning.reasoning_content
        content = reasoning.content
        if FALLBACK_TO_REASONING and content is None and reasoning.reasoning_content and not parsed.tool_calls:
            content = reasoning.reasoning_content
        message["content"] = content

        if parsed.tool_calls:
            message["tool_calls"] = parsed.tool_calls
            choice["finish_reason"] = "tool_calls"
        choice["message"] = message

        if LOG_COMPLETIONS:
            print(
                "normalized choice "
                f"index={choice.get('index', 0)} "
                f"finish={choice.get('finish_reason')} "
                f"content_len={len(message.get('content') or '')} "
                f"reasoning_len={len(message.get('reasoning_content') or '')} "
                f"tool_calls={len(message.get('tool_calls') or [])}",
                flush=True,
            )

    return response


def stream_buffered_completion(payload: dict[str, Any]) -> Iterable[bytes]:
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            result_queue.put(("ok", normalize_completion(create_completion(payload, stream=False))))
        except BaseException as exc:  # noqa: BLE001 - surfaced to the SSE client below.
            result_queue.put(("error", exc))

    thread = threading.Thread(target=worker, name="vibethinker-generation", daemon=True)
    thread.start()

    yield _sse(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
    )

    while True:
        try:
            status, result = result_queue.get(timeout=10)
            break
        except queue.Empty:
            yield b": keep-alive\n\n"

    if status == "error":
        yield _sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": f"\n[local gateway error: {type(result).__name__}: {result}]"
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        yield b"data: [DONE]\n\n"
        return

    normalized = result
    normalized["id"] = completion_id

    for choice in normalized.get("choices", []):
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls")
        content = message.get("content")

        if tool_calls:
            yield _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": choice.get("index", 0),
                            "delta": {"tool_calls": tool_calls},
                            "finish_reason": None,
                        }
                    ],
                }
            )
            finish_reason = "tool_calls"
        else:
            yield _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": choice.get("index", 0),
                            "delta": {"content": content or ""},
                            "finish_reason": None,
                        }
                    ],
                }
            )
            finish_reason = choice.get("finish_reason") or "stop"

        yield _sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": choice.get("index", 0),
                        "delta": {},
                        "finish_reason": finish_reason,
                    }
                ],
            }
        )

    yield b"data: [DONE]\n\n"


def stream_native_completion(chunks: Iterable[dict[str, Any]]) -> Iterable[bytes]:
    for chunk in chunks:
        if isinstance(chunk, dict):
            chunk["model"] = MODEL_ID
        yield _sse(chunk)
    yield b"data: [DONE]\n\n"


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
