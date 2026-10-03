from pydantic import BaseModel

from secops.tools.base import ToolSpec


class _In(BaseModel):
    x: int


class _Out(BaseModel):
    y: int
    source: str = "test"


def test_toolspec_is_read_only_and_json_schema_able() -> None:
    spec = ToolSpec(
        name="double",
        description="Doubles x. Cannot do anything else.",
        input_model=_In,
        output_model=_Out,
        run=lambda inp: _Out(y=inp.x * 2),  # type: ignore[attr-defined]
        external_source=None,
    )
    assert spec.read_only is True
    assert spec.input_schema()["properties"]["x"]["type"] == "integer"
    assert spec.invoke({"x": 4}).y == 8  # type: ignore[attr-defined]
