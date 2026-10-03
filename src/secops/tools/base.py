"""Tool contract shared by every tool: typed input, typed output, read-only, bounded."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class ToolSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    run: Callable[[Any], BaseModel]
    read_only: Literal[True] = True
    external_source: str | None = None

    def input_schema(self) -> dict[str, Any]:
        return self.input_model.model_json_schema()

    def output_schema(self) -> dict[str, Any]:
        return self.output_model.model_json_schema()

    def invoke(self, raw: dict[str, Any]) -> BaseModel:
        """Validate raw arguments (as an LLM would supply them) and run the tool."""
        return self.run(self.input_model.model_validate(raw))
