from local_opencode_gateway.tool_calls import parse_reasoning, parse_tool_calls


def test_parse_single_tool_call() -> None:
    parsed = parse_tool_calls(
        'I will inspect it.\n<tool_call>\n{"name":"bash","arguments":{"cmd":"pwd"}}\n</tool_call>'
    )

    assert parsed.content == "I will inspect it."
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["type"] == "function"
    assert parsed.tool_calls[0]["function"]["name"] == "bash"
    assert parsed.tool_calls[0]["function"]["arguments"] == '{"cmd":"pwd"}'


def test_parse_multiple_tool_calls_and_empty_content() -> None:
    parsed = parse_tool_calls(
        '<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>\n'
        '<tool_call>{"name":"list","arguments":{}}</tool_call>'
    )

    assert parsed.content is None
    assert [call["function"]["name"] for call in parsed.tool_calls] == ["read", "list"]


def test_invalid_json_is_left_as_content() -> None:
    text = "<tool_call>{bad json}</tool_call>"
    parsed = parse_tool_calls(text)

    assert parsed.content == text
    assert parsed.tool_calls == []


def test_parse_reasoning() -> None:
    parsed = parse_reasoning("<think>scratch work</think>126")

    assert parsed.reasoning_content == "scratch work"
    assert parsed.content == "126"


def test_parse_truncated_reasoning() -> None:
    parsed = parse_reasoning("<think>unfinished scratch work")

    assert parsed.reasoning_content == "unfinished scratch work"
    assert parsed.content is None
