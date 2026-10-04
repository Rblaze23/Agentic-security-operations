"""Record and replay tool outputs so a recorded scenario replays without the event store.

The LLM adapter replays model responses by request hash; the next request embeds the tool
results, so those must be byte-identical too. `recording_registry` saves every output the real
tools produce; `replay_registry` serves them back keyed by tool name and validated arguments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from secops.tools.base import ToolSpec
from secops.tools.registry import ToolRegistry


class UnrecordedToolCallError(RuntimeError):
    """A replayed scenario asked a tool something the recording never asked."""


def call_key(name: str, inp: BaseModel) -> str:
    args = json.dumps(inp.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha256(f"{name}\n{args}".encode()).hexdigest()


def recording_registry(base: ToolRegistry, directory: Path) -> ToolRegistry:
    directory.mkdir(parents=True, exist_ok=True)
    counter = {"n": len(list(directory.glob("*.json")))}
    out = ToolRegistry()
    for spec in base.all():
        out.register(_recording_spec(spec, directory, counter))
    return out


def _recording_spec(spec: ToolSpec, directory: Path, counter: dict[str, int]) -> ToolSpec:
    real = spec.run

    def run(inp: BaseModel) -> BaseModel:
        result = real(inp)
        counter["n"] += 1
        (directory / f"{counter['n']:03d}_{spec.name}.json").write_text(
            json.dumps(
                {
                    "key": call_key(spec.name, inp),
                    "tool": spec.name,
                    "arguments": inp.model_dump(mode="json"),
                    "output": result.model_dump(mode="json"),
                },
                indent=2,
                default=str,
            )
        )
        return result

    return spec.model_copy(update={"run": run})


def replay_registry(base: ToolRegistry, directory: Path) -> ToolRegistry:
    """`base` supplies the input/output models (any registry built from the same code)."""
    recorded: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        entry = json.loads(path.read_text())
        recorded.setdefault(entry["key"], entry)  # first occurrence wins, like a cache
    out = ToolRegistry()
    for spec in base.all():
        out.register(_replay_spec(spec, recorded))
    return out


def _replay_spec(spec: ToolSpec, recorded: dict[str, dict[str, Any]]) -> ToolSpec:
    def run(inp: BaseModel) -> BaseModel:
        key = call_key(spec.name, inp)
        entry = recorded.get(key)
        if entry is None:
            raise UnrecordedToolCallError(
                f"{spec.name} called with unrecorded arguments {inp.model_dump(mode='json')}"
            )
        return spec.output_model.model_validate(entry["output"])

    return spec.model_copy(update={"run": run})
