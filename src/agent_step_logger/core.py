"""Structured per-step logging for AI agent loops.

Each step in an agent run (tool call, tool result, thinking, response, error)
is recorded as a :class:`Step` with a type, content, and timestamp.  The log
can be serialised to / from JSONL for persistence and later replay or analysis.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


class StepType(str, Enum):
    """Canonical step types emitted during an agent run."""

    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    RESPONSE = "response"
    ERROR = "error"
    CUSTOM = "custom"


@dataclass
class Step:
    """A single recorded step in an agent run.

    Attributes:
        index: Zero-based position in the log.
        step_type: The kind of step.
        content: Human-readable or structured content for this step.
        metadata: Optional extra key/value pairs attached to the step.
        timestamp: Unix timestamp (seconds) when the step was recorded.
    """

    index: int
    step_type: StepType
    content: str
    metadata: dict[str, Any]
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise this step to a plain dict."""
        return {
            "index": self.index,
            "step_type": self.step_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step:
        """Reconstruct a :class:`Step` from a plain dict."""
        return cls(
            index=int(data["index"]),
            step_type=StepType(data["step_type"]),
            content=str(data["content"]),
            metadata=dict(data.get("metadata") or {}),
            timestamp=float(data["timestamp"]),
        )

    def __repr__(self) -> str:
        preview = self.content[:40]
        suffix = "…" if len(self.content) > 40 else ""
        return (
            f"Step(index={self.index}, type={self.step_type.value!r},"
            f" content={preview + suffix!r})"
        )


class StepLogger:
    """Record agent steps and query or serialise them.

    Args:
        clock: Optional callable that returns the current time as a float
            (seconds).  Defaults to :func:`time.time`.  Useful for testing.

    Example::

        logger = StepLogger()
        logger.log_tool_call("search", {"query": "python slots"})
        logger.log_tool_result("search", "Found 42 results.")
        logger.log_response("Here is the answer…")
        print(logger.summary())
    """

    def __init__(self, *, clock: Any = None) -> None:
        self._steps: list[Step] = []
        self._clock = clock if clock is not None else time.time

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _add(
        self,
        step_type: StepType,
        content: str,
        metadata: dict[str, Any] | None,
    ) -> Step:
        step = Step(
            index=len(self._steps),
            step_type=step_type,
            content=str(content),
            metadata=dict(metadata) if metadata else {},
            timestamp=self._clock(),
        )
        self._steps.append(step)
        return step

    def log_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        *,
        extra: dict[str, Any] | None = None,
    ) -> Step:
        """Record a tool invocation.

        Args:
            tool_name: Name of the tool being called.
            args: Arguments passed to the tool.
            extra: Additional metadata to attach.

        Returns:
            The recorded :class:`Step`.
        """
        meta: dict[str, Any] = {"tool_name": tool_name}
        if args is not None:
            meta["args"] = args
        if extra:
            meta.update(extra)
        return self._add(StepType.TOOL_CALL, tool_name, meta)

    def log_tool_result(
        self,
        tool_name: str,
        result: Any,
        *,
        extra: dict[str, Any] | None = None,
    ) -> Step:
        """Record the result returned by a tool.

        Args:
            tool_name: Name of the tool that was called.
            result: The tool's return value (converted to str for content).
            extra: Additional metadata to attach.

        Returns:
            The recorded :class:`Step`.
        """
        meta: dict[str, Any] = {"tool_name": tool_name}
        if extra:
            meta.update(extra)
        return self._add(StepType.TOOL_RESULT, str(result), meta)

    def log_thinking(
        self,
        text: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> Step:
        """Record an internal reasoning / thinking block.

        Args:
            text: The thinking text.
            extra: Additional metadata to attach.

        Returns:
            The recorded :class:`Step`.
        """
        return self._add(StepType.THINKING, text, extra)

    def log_response(
        self,
        text: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> Step:
        """Record a final or intermediate response from the model.

        Args:
            text: The response text.
            extra: Additional metadata to attach.

        Returns:
            The recorded :class:`Step`.
        """
        return self._add(StepType.RESPONSE, text, extra)

    def log_error(
        self,
        error: str | Exception,
        *,
        extra: dict[str, Any] | None = None,
    ) -> Step:
        """Record an error that occurred during the agent run.

        Args:
            error: The error message or exception.
            extra: Additional metadata to attach.

        Returns:
            The recorded :class:`Step`.
        """
        meta: dict[str, Any] = {}
        if isinstance(error, Exception):
            meta["error_type"] = type(error).__name__
        if extra:
            meta.update(extra)
        return self._add(StepType.ERROR, str(error), meta)

    def log_custom(
        self,
        label: str,
        content: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> Step:
        """Record a custom step not covered by the standard types.

        Args:
            label: Short label stored in metadata as ``"label"``.
            content: Step content.
            extra: Additional metadata to attach.

        Returns:
            The recorded :class:`Step`.
        """
        meta: dict[str, Any] = {"label": label}
        if extra:
            meta.update(extra)
        return self._add(StepType.CUSTOM, content, meta)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def steps(self) -> list[Step]:
        """Return a copy of all recorded steps."""
        return list(self._steps)

    def filter(self, step_type: StepType | str) -> list[Step]:
        """Return only steps matching *step_type*.

        Args:
            step_type: A :class:`StepType` value or its string equivalent.

        Returns:
            Filtered list of :class:`Step` objects.
        """
        target = StepType(step_type) if isinstance(step_type, str) else step_type
        return [s for s in self._steps if s.step_type == target]

    def __len__(self) -> int:
        return len(self._steps)

    def __getitem__(self, index: int) -> Step:
        return self._steps[index]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a counts-by-type summary dict.

        Returns:
            Dict with ``"total"`` and one key per :class:`StepType` value
            that appears in the log (counts are omitted when zero).
        """
        counts: dict[str, int] = {}
        for step in self._steps:
            counts[step.step_type.value] = counts.get(step.step_type.value, 0) + 1
        return {"total": len(self._steps), **counts}

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_jsonl(self) -> str:
        """Serialise all steps to a JSONL string (one JSON object per line).

        Returns:
            Multi-line string; empty string if there are no steps.
        """
        if not self._steps:
            return ""
        lines = [
            json.dumps(step.to_dict(), separators=(",", ":")) for step in self._steps
        ]
        return "\n".join(lines)

    @classmethod
    def from_jsonl(cls, text: str, *, clock: Any = None) -> StepLogger:
        """Reconstruct a :class:`StepLogger` from a JSONL string.

        Args:
            text: JSONL string produced by :meth:`to_jsonl`.
            clock: Optional clock callable for the new instance.

        Returns:
            A new :class:`StepLogger` pre-populated with the decoded steps.
        """
        logger = cls(clock=clock)
        for line in text.splitlines():
            line = line.strip()
            if line:
                logger._steps.append(Step.from_dict(json.loads(line)))
        return logger

    def clear(self) -> None:
        """Remove all recorded steps."""
        self._steps.clear()

    def __repr__(self) -> str:
        return f"StepLogger(steps={len(self._steps)})"
