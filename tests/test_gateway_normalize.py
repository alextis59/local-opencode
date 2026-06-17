import local_opencode_gateway.gateway as gateway
from local_opencode_gateway.gateway import build_completion_kwargs, normalize_completion


def test_normalize_completion_falls_back_to_reasoning_content() -> None:
    completion = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "raw",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "<think>still reasoning",
                },
                "finish_reason": "length",
            }
        ],
    }

    normalized = normalize_completion(completion)
    message = normalized["choices"][0]["message"]

    assert message["content"] == "still reasoning"
    assert message["reasoning_content"] == "still reasoning"


def test_build_completion_kwargs_caps_tokens_and_omits_tools() -> None:
    kwargs = build_completion_kwargs(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10_000,
            "tools": [{"type": "function", "function": {"name": "x"}}],
            "tool_choice": "auto",
        },
        stream=True,
    )

    assert kwargs["max_tokens"] == 64
    assert kwargs["stream"] is True
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs


def test_filtered_tools_uses_allowlist(monkeypatch) -> None:
    monkeypatch.setattr(gateway, "FORWARD_TOOL_NAMES", {"write", "bash"})

    tools = gateway.filtered_tools(
        [
            {"type": "function", "function": {"name": "read"}},
            {"type": "function", "function": {"name": "write"}},
            {"type": "function", "function": {"name": "bash"}},
        ]
    )

    assert [tool["function"]["name"] for tool in tools] == ["write", "bash"]


def test_build_completion_kwargs_can_force_response_format(monkeypatch) -> None:
    monkeypatch.setattr(gateway, "RESPONSE_FORMAT", "json_object")

    kwargs = build_completion_kwargs({"messages": [{"role": "user", "content": "hi"}]}, stream=False)

    assert kwargs["response_format"] == {"type": "json_object"}
