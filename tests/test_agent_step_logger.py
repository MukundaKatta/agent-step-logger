"""Tests for agent-step-logger."""

from __future__ import annotations

import json

import pytest

from agent_step_logger import Step, StepLogger, StepType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_clock(start: float = 0.0):
    t = [start]

    def clock():
        return t[0]

    def advance(s: float):
        t[0] += s

    return clock, advance


# ---------------------------------------------------------------------------
# StepType enum
# ---------------------------------------------------------------------------


def test_step_type_values():
    assert StepType.TOOL_CALL.value == "tool_call"
    assert StepType.TOOL_RESULT.value == "tool_result"
    assert StepType.THINKING.value == "thinking"
    assert StepType.RESPONSE.value == "response"
    assert StepType.ERROR.value == "error"
    assert StepType.CUSTOM.value == "custom"


def test_step_type_from_string():
    assert StepType("tool_call") is StepType.TOOL_CALL


# ---------------------------------------------------------------------------
# Step dataclass
# ---------------------------------------------------------------------------


def test_step_to_dict():
    s = Step(
        index=0, step_type=StepType.RESPONSE, content="hi", metadata={}, timestamp=1.0
    )
    d = s.to_dict()
    assert d["index"] == 0
    assert d["step_type"] == "response"
    assert d["content"] == "hi"
    assert d["timestamp"] == 1.0


def test_step_from_dict():
    d = {
        "index": 2,
        "step_type": "error",
        "content": "oops",
        "metadata": {"k": "v"},
        "timestamp": 5.0,
    }
    s = Step.from_dict(d)
    assert s.index == 2
    assert s.step_type == StepType.ERROR
    assert s.content == "oops"
    assert s.metadata == {"k": "v"}
    assert s.timestamp == 5.0


def test_step_repr_short():
    s = Step(
        index=0,
        step_type=StepType.THINKING,
        content="short",
        metadata={},
        timestamp=0.0,
    )
    r = repr(s)
    assert "thinking" in r
    assert "short" in r
    assert "…" not in r


def test_step_repr_long():
    long_content = "x" * 80
    s = Step(
        index=0,
        step_type=StepType.THINKING,
        content=long_content,
        metadata={},
        timestamp=0.0,
    )
    r = repr(s)
    assert "…" in r


# ---------------------------------------------------------------------------
# StepLogger construction
# ---------------------------------------------------------------------------


def test_default_logger():
    logger = StepLogger()
    assert len(logger) == 0


def test_repr():
    logger = StepLogger()
    assert "StepLogger" in repr(logger)
    assert "0" in repr(logger)


# ---------------------------------------------------------------------------
# log_tool_call
# ---------------------------------------------------------------------------


def test_log_tool_call_basic():
    logger = StepLogger()
    step = logger.log_tool_call("search")
    assert step.step_type == StepType.TOOL_CALL
    assert step.content == "search"
    assert step.metadata["tool_name"] == "search"


def test_log_tool_call_with_args():
    logger = StepLogger()
    step = logger.log_tool_call("search", {"query": "python"})
    assert step.metadata["args"] == {"query": "python"}


def test_log_tool_call_with_extra():
    logger = StepLogger()
    step = logger.log_tool_call("search", extra={"run_id": "abc"})
    assert step.metadata["run_id"] == "abc"


# ---------------------------------------------------------------------------
# log_tool_result
# ---------------------------------------------------------------------------


def test_log_tool_result_basic():
    logger = StepLogger()
    step = logger.log_tool_result("search", "42 results")
    assert step.step_type == StepType.TOOL_RESULT
    assert step.content == "42 results"
    assert step.metadata["tool_name"] == "search"


def test_log_tool_result_non_string():
    logger = StepLogger()
    step = logger.log_tool_result("calc", 99)
    assert step.content == "99"


def test_log_tool_result_with_extra():
    logger = StepLogger()
    step = logger.log_tool_result("calc", 1, extra={"cached": True})
    assert step.metadata["cached"] is True


# ---------------------------------------------------------------------------
# log_thinking
# ---------------------------------------------------------------------------


def test_log_thinking():
    logger = StepLogger()
    step = logger.log_thinking("let me think…")
    assert step.step_type == StepType.THINKING
    assert step.content == "let me think…"


def test_log_thinking_with_extra():
    logger = StepLogger()
    step = logger.log_thinking("hmm", extra={"model": "claude"})
    assert step.metadata["model"] == "claude"


# ---------------------------------------------------------------------------
# log_response
# ---------------------------------------------------------------------------


def test_log_response():
    logger = StepLogger()
    step = logger.log_response("Here is the answer.")
    assert step.step_type == StepType.RESPONSE
    assert step.content == "Here is the answer."


def test_log_response_with_extra():
    logger = StepLogger()
    step = logger.log_response("ok", extra={"tokens": 5})
    assert step.metadata["tokens"] == 5


# ---------------------------------------------------------------------------
# log_error
# ---------------------------------------------------------------------------


def test_log_error_string():
    logger = StepLogger()
    step = logger.log_error("something went wrong")
    assert step.step_type == StepType.ERROR
    assert step.content == "something went wrong"


def test_log_error_exception():
    logger = StepLogger()
    err = ValueError("bad value")
    step = logger.log_error(err)
    assert step.content == "bad value"
    assert step.metadata["error_type"] == "ValueError"


def test_log_error_with_extra():
    logger = StepLogger()
    step = logger.log_error("oops", extra={"retry": True})
    assert step.metadata["retry"] is True


# ---------------------------------------------------------------------------
# log_custom
# ---------------------------------------------------------------------------


def test_log_custom():
    logger = StepLogger()
    step = logger.log_custom("checkpoint", "saved state")
    assert step.step_type == StepType.CUSTOM
    assert step.content == "saved state"
    assert step.metadata["label"] == "checkpoint"


def test_log_custom_with_extra():
    logger = StepLogger()
    step = logger.log_custom("note", "blah", extra={"priority": 1})
    assert step.metadata["priority"] == 1


# ---------------------------------------------------------------------------
# Index and ordering
# ---------------------------------------------------------------------------


def test_step_indices():
    logger = StepLogger()
    logger.log_tool_call("a")
    logger.log_tool_result("a", "ok")
    logger.log_response("done")
    steps = logger.steps()
    assert [s.index for s in steps] == [0, 1, 2]


def test_getitem():
    logger = StepLogger()
    logger.log_response("hello")
    assert logger[0].content == "hello"


def test_len():
    logger = StepLogger()
    assert len(logger) == 0
    logger.log_response("x")
    assert len(logger) == 1


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


def test_filter_by_type():
    logger = StepLogger()
    logger.log_tool_call("a")
    logger.log_response("r1")
    logger.log_response("r2")
    responses = logger.filter(StepType.RESPONSE)
    assert len(responses) == 2


def test_filter_by_string():
    logger = StepLogger()
    logger.log_thinking("t")
    logger.log_response("r")
    assert len(logger.filter("thinking")) == 1


def test_filter_empty():
    logger = StepLogger()
    logger.log_response("r")
    assert logger.filter(StepType.ERROR) == []


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def test_summary_empty():
    logger = StepLogger()
    s = logger.summary()
    assert s["total"] == 0
    assert "response" not in s


def test_summary_counts():
    logger = StepLogger()
    logger.log_tool_call("a")
    logger.log_tool_call("b")
    logger.log_response("r")
    s = logger.summary()
    assert s["total"] == 3
    assert s["tool_call"] == 2
    assert s["response"] == 1


def test_summary_only_present_types():
    logger = StepLogger()
    logger.log_response("r")
    s = logger.summary()
    assert "error" not in s


# ---------------------------------------------------------------------------
# steps() returns copy
# ---------------------------------------------------------------------------


def test_steps_returns_copy():
    logger = StepLogger()
    logger.log_response("r")
    copy = logger.steps()
    copy.clear()
    assert len(logger) == 1


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


def test_clear():
    logger = StepLogger()
    logger.log_response("r")
    logger.clear()
    assert len(logger) == 0


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def test_timestamps_monotonic():
    clock, advance = make_clock(100.0)
    logger = StepLogger(clock=clock)
    logger.log_response("a")
    advance(1.5)
    logger.log_response("b")
    steps = logger.steps()
    assert steps[0].timestamp == pytest.approx(100.0)
    assert steps[1].timestamp == pytest.approx(101.5)


# ---------------------------------------------------------------------------
# to_jsonl / from_jsonl
# ---------------------------------------------------------------------------


def test_to_jsonl_empty():
    logger = StepLogger()
    assert logger.to_jsonl() == ""


def test_to_jsonl_single():
    clock, _ = make_clock(1.0)
    logger = StepLogger(clock=clock)
    logger.log_response("hello")
    jsonl = logger.to_jsonl()
    lines = jsonl.splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["content"] == "hello"
    assert obj["step_type"] == "response"


def test_to_jsonl_multiple():
    logger = StepLogger()
    logger.log_tool_call("a")
    logger.log_tool_result("a", "ok")
    jsonl = logger.to_jsonl()
    assert len(jsonl.splitlines()) == 2


def test_from_jsonl_roundtrip():
    clock, advance = make_clock(0.0)
    logger = StepLogger(clock=clock)
    logger.log_tool_call("search", {"q": "hi"})
    advance(0.5)
    logger.log_tool_result("search", "found 3")
    advance(0.5)
    logger.log_response("Here you go.")

    jsonl = logger.to_jsonl()
    restored = StepLogger.from_jsonl(jsonl)

    assert len(restored) == 3
    assert restored[0].step_type == StepType.TOOL_CALL
    assert restored[1].step_type == StepType.TOOL_RESULT
    assert restored[2].content == "Here you go."


def test_from_jsonl_preserves_metadata():
    logger = StepLogger()
    logger.log_tool_call("calc", {"x": 1}, extra={"retry": False})
    restored = StepLogger.from_jsonl(logger.to_jsonl())
    assert restored[0].metadata["tool_name"] == "calc"
    assert restored[0].metadata["retry"] is False


def test_from_jsonl_ignores_blank_lines():
    logger = StepLogger()
    logger.log_response("r")
    jsonl = "\n" + logger.to_jsonl() + "\n\n"
    restored = StepLogger.from_jsonl(jsonl)
    assert len(restored) == 1


def test_from_jsonl_preserves_indices():
    logger = StepLogger()
    logger.log_response("r0")
    logger.log_response("r1")
    restored = StepLogger.from_jsonl(logger.to_jsonl())
    assert restored[0].index == 0
    assert restored[1].index == 1


# ---------------------------------------------------------------------------
# Step.to_dict / from_dict symmetry
# ---------------------------------------------------------------------------


def test_step_dict_roundtrip_equal():
    s = Step(
        index=3,
        step_type=StepType.TOOL_CALL,
        content="search",
        metadata={"tool_name": "search", "args": {"q": "x"}},
        timestamp=12.5,
    )
    assert Step.from_dict(s.to_dict()) == s


def test_step_from_dict_defaults_missing_metadata():
    d = {
        "index": 0,
        "step_type": "response",
        "content": "hi",
        "timestamp": 1.0,
    }
    s = Step.from_dict(d)
    assert s.metadata == {}


# ---------------------------------------------------------------------------
# __getitem__ negative index
# ---------------------------------------------------------------------------


def test_getitem_negative_index():
    logger = StepLogger()
    logger.log_response("first")
    logger.log_response("last")
    assert logger[-1].content == "last"
