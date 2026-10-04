"""Replay the recorded scenarios (zero cost) and rewrite report.json / result.json.

Use after a change to the finalize step (severity rubric, family normalisation) that does not
touch any model request: the model fixtures stay valid and only the derived report changes.
Usage and cost fields are kept from the original recording."""

from __future__ import annotations

import json
from pathlib import Path

from secops.agent.graph import AgentDeps, run_investigation
from secops.agent.llm import LLM
from secops.agent.replay import replay_registry
from secops.schemas.alert import Alert
from secops.tools.registry import build_registry

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "llm"


def main() -> None:
    base = build_registry(include_detector=True)
    for scenario_dir in sorted(p for p in FIXTURE_ROOT.iterdir() if (p / "result.json").exists()):
        name = scenario_dir.name
        alert = Alert.model_validate_json((scenario_dir / "alert.json").read_text())
        saved = json.loads((scenario_dir / "result.json").read_text())
        deps = AgentDeps(
            registry=replay_registry(base, scenario_dir / "tools"),
            investigator=LLM(
                "claude-opus-5-5", mode="replay", fixture_dir=scenario_dir / "investigator"
            ),
            critic=LLM("claude-sonnet-5-5", mode="replay", fixture_dir=scenario_dir / "critic"),
            tool_budget=12,
        )
        result = run_investigation(alert, deps, investigation_id=name)
        changed = {
            k: (saved[k], v)
            for k, v in {
                "verdict": result.report.verdict,
                "severity": result.report.severity,
                "attack_family": result.report.attack_family,
            }.items()
            if saved[k] != v
        }
        saved.update(
            verdict=result.report.verdict,
            severity=result.report.severity,
            attack_family=result.report.attack_family,
            confidence=result.report.confidence,
            status=result.status,
            tool_calls=[c.tool for c in result.tool_calls],
            critic_rejections=result.iteration,
            issues=[i.model_dump() for i in result.all_issues],
            prompt_version=result.prompt_version,
            errors=result.errors,
        )
        (scenario_dir / "result.json").write_text(json.dumps(saved, indent=2, default=str))
        (scenario_dir / "report.json").write_text(result.report.model_dump_json(indent=2))
        print(f"{name}: {'unchanged' if not changed else changed}")


if __name__ == "__main__":
    main()
