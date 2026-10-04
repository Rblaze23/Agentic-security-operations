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

from secops.agent.fixtures import count_entries, read_entries, write_entry
from secops.tools.base import ToolSpec
from secops.tools.registry import ToolRegistry


class UnrecordedToolCallError(RuntimeError):
    """A replayed scenario asked a tool something the recording never asked."""


def call_key(name: str, inp: BaseModel) -> str:
    args = json.dumps(inp.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha256(f"{name}\n{args}".encode()).hexdigest()


def recording_registry(base: ToolRegistry, directory: Path, compress: bool = False) -> ToolRegistry:
    directory.mkdir(parents=True, exist_ok=True)
    counter = {"n": count_entries(directory)}
    out = ToolRegistry()
    for spec in base.all():
        out.register(_recording_spec(spec, directory, counter, compress))
    return out


def _recording_spec(
    spec: ToolSpec, directory: Path, counter: dict[str, int], compress: bool
) -> ToolSpec:
    real = spec.run

    def run(inp: BaseModel) -> BaseModel:
        result = real(inp)
        counter["n"] += 1
        write_entry(
            directory,
            f"{counter['n']:03d}_{spec.name}",
            {
                "key": call_key(spec.name, inp),
                "tool": spec.name,
                "arguments": inp.model_dump(mode="json"),
                "output": result.model_dump(mode="json"),
            },
            compress=compress,
        )
        return result

    return spec.model_copy(update={"run": run})


def replay_registry(base: ToolRegistry, directory: Path) -> ToolRegistry:
    """`base` supplies the input/output models (any registry built from the same code)."""
    recorded: dict[str, dict[str, Any]] = {}
    for entry in read_entries(directory):
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
