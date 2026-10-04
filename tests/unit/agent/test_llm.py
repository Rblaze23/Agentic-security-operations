import json
from pathlib import Path
from typing import Any

import pytest
from anthropic.types import Message
from pydantic import BaseModel

from secops.agent.llm import (
    LLM,
    LLMRefusalError,
    LLMTruncatedError,
    UnrecordedRequestError,
    request_fingerprint,
)
from secops.schemas.agent import UsageTotals


class Answer(BaseModel):
    verdict: str
    score: float


def _message(
    content: list[dict[str, Any]],
    stop_reason: str = "end_turn",
    usage: dict[str, int] | None = None,
    stop_details: dict[str, Any] | None = None,
    model: str = "claude-opus-5-5",
) -> Message:
    payload: dict[str, Any] = {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 1200,
            "output_tokens": 300,
            "cache_read_input_tokens": 1000,
            "cache_creation_input_tokens": 200,
            **(usage or {}),
        },
    }
    if stop_details is not None:
        payload["stop_details"] = stop_details
    return Message.model_validate(payload)


class FakeMessages:
    def __init__(self, responses: list[Message]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Message:
        self.requests.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[Message]) -> None:
        self.messages = FakeMessages(responses)


def test_live_call_builds_request_with_caching_effort_and_format() -> None:
    client = FakeClient([_message([{"type": "text", "text": '{"verdict": "ok", "score": 0.5}'}])])
    llm = LLM(model="claude-opus-5-5", effort="low", mode="live", client=client)
    tools = [
        {"name": "t", "description": "d", "input_schema": {"type": "object", "properties": {}}}
    ]
    r = llm.create(
        system="SYS", messages=[{"role": "user", "content": "hi"}], tools=tools, output_model=Answer
    )
    req = client.messages.requests[0]
    assert req["model"] == "claude-opus-5-5" and req["max_tokens"] >= 4000
    assert req["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert req["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert req["output_config"]["effort"] == "low"
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert req["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert "thinking" not in req and "tool_choice" not in req
    assert r.parsed == {"verdict": "ok", "score": 0.5}
    assert r.usage.input_tokens == 1200 and r.usage.cache_read_tokens == 1000
    assert llm.usage.cost_usd == pytest.approx((1200 * 4 + 300 * 20 + 200 * 5 + 1000 * 0.2) / 1e6)


def test_record_then_replay_serves_the_same_response(tmp_path: Path) -> None:
    content = [{"type": "text", "text": '{"verdict": "tp", "score": 0.9}'}]
    recorder = LLM(
        model="claude-opus-5-5",
        mode="record",
        fixture_dir=tmp_path,
        client=FakeClient([_message(content)]),
    )
    first = recorder.create(
        system="S", messages=[{"role": "user", "content": "q"}], output_model=Answer
    )
    files = sorted(tmp_path.glob("*.json"))
    assert len(files) == 1
    stored = json.loads(files[0].read_text())
    assert stored["request_hash"] == request_fingerprint(stored["request"])
    assert stored["model"] == "claude-opus-5-5" and "captured_at" in stored

    replayer = LLM(model="claude-opus-5-5", mode="replay", fixture_dir=tmp_path)  # no client
    again = replayer.create(
        system="S", messages=[{"role": "user", "content": "q"}], output_model=Answer
    )
    assert again.parsed == first.parsed and again.content == first.content
    assert replayer.usage.cost_usd == pytest.approx(recorder.usage.cost_usd)


def test_replay_is_deterministic_and_rejects_unseen_requests(tmp_path: Path) -> None:
    content = [{"type": "text", "text": '{"verdict": "fp", "score": 0.1}'}]
    LLM(
        model="claude-opus-5-5",
        mode="record",
        fixture_dir=tmp_path,
        client=FakeClient([_message(content)]),
    ).create(system="S", messages=[{"role": "user", "content": "q"}], output_model=Answer)
    a = LLM(model="claude-opus-5-5", mode="replay", fixture_dir=tmp_path)
    b = LLM(model="claude-opus-5-5", mode="replay", fixture_dir=tmp_path)
    ra = a.create(system="S", messages=[{"role": "user", "content": "q"}], output_model=Answer)
    rb = b.create(system="S", messages=[{"role": "user", "content": "q"}], output_model=Answer)
    assert ra.model_dump() == rb.model_dump()
    with pytest.raises(UnrecordedRequestError, match="prompt changed"):
        a.create(
            system="S changed", messages=[{"role": "user", "content": "q"}], output_model=Answer
        )


def test_refusal_and_truncation_become_typed_errors() -> None:
    refusing = FakeClient(
        [
            _message(
                [],
                stop_reason="refusal",
                stop_details={"type": "refusal", "category": "cyber", "explanation": "no"},
            )
        ]
    )
    with pytest.raises(LLMRefusalError) as exc:
        LLM(model="claude-opus-5-5", mode="live", client=refusing).create(
            system="S", messages=[{"role": "user", "content": "q"}]
        )
    assert exc.value.category == "cyber"
    truncated = FakeClient(
        [_message([{"type": "text", "text": "partial"}], stop_reason="max_tokens")]
    )
    with pytest.raises(LLMTruncatedError):
        LLM(model="claude-opus-5-5", mode="live", client=truncated).create(
            system="S", messages=[{"role": "user", "content": "q"}]
        )


def test_tool_use_blocks_are_preserved_and_usage_shared() -> None:
    shared = UsageTotals()
    client = FakeClient(
        [
            _message(
                [
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "search_events",
                        "input": {"limit": 5},
                    }
                ],
                stop_reason="tool_use",
                model="claude-sonnet-5-5",
            )
        ]
    )
    llm = LLM(model="claude-sonnet-5-5", mode="live", client=client, usage=shared)
    r = llm.create(
        system="S",
        messages=[{"role": "user", "content": "q"}],
        tools=[{"name": "search_events", "description": "d", "input_schema": {"type": "object"}}],
    )
    assert (
        r.stop_reason == "tool_use"
        and r.tool_uses[0]["name"] == "search_events"
        and r.tool_uses[0]["input"] == {"limit": 5}
    )
    assert r.assistant_message == {"role": "assistant", "content": r.content}
    assert shared.by_model["claude-sonnet-5-5"].calls == 1


def test_unknown_mode_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        LLM(model="claude-opus-5-5", mode="dryrun")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fixture_dir"):
        LLM(model="claude-opus-5-5", mode="replay")
