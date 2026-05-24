# agent-step-logger

Structured per-step logging for AI agent loops.

Records every step of an agent run — tool calls, tool results, thinking blocks, model responses, and errors — as typed `Step` objects. Steps can be queried, filtered by type, and serialised to JSONL for persistence or later replay.

## Install

```bash
pip install agent-step-logger
```

## Quick start

```python
from agent_step_logger import StepLogger

logger = StepLogger()

logger.log_tool_call("search", {"query": "python slots"})
logger.log_tool_result("search", "Found 42 results.")
logger.log_thinking("The user wants info about Python __slots__.")
logger.log_response("Here is what I found about Python __slots__…")

print(logger.summary())
# {'total': 4, 'tool_call': 1, 'tool_result': 1, 'thinking': 1, 'response': 1}
```

## API

### `StepLogger(*, clock=None)`

Create a logger. Pass a custom `clock` callable (returns float seconds) for testing.

### Logging methods

| Method | StepType |
|---|---|
| `log_tool_call(tool_name, args=None, *, extra=None)` | `tool_call` |
| `log_tool_result(tool_name, result, *, extra=None)` | `tool_result` |
| `log_thinking(text, *, extra=None)` | `thinking` |
| `log_response(text, *, extra=None)` | `response` |
| `log_error(error, *, extra=None)` | `error` |
| `log_custom(label, content, *, extra=None)` | `custom` |

All methods return the recorded `Step` and accept an optional `extra` dict merged into `metadata`.

### Querying

```python
logger.steps()              # list[Step] — all steps (copy)
logger.filter(StepType.TOOL_CALL)   # list[Step] — by type
logger[0]                   # Step at index
len(logger)                 # total step count
logger.summary()            # {'total': N, 'tool_call': 2, ...}
logger.clear()              # remove all steps
```

### Serialisation

```python
jsonl = logger.to_jsonl()               # multi-line string
restored = StepLogger.from_jsonl(jsonl) # reconstructed logger
```

### `Step`

```python
step.index       # int — position in log
step.step_type   # StepType
step.content     # str
step.metadata    # dict[str, Any]
step.timestamp   # float (Unix seconds)
step.to_dict()   # serialise
Step.from_dict(d) # deserialise
```

## License

MIT
