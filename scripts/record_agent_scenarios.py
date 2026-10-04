"""Record the four real agent scenarios as replay fixtures, printing the measured cost.

Usage:
    uv run python scripts/record_agent_scenarios.py            # dry run: build and show the alerts
    uv run python scripts/record_agent_scenarios.py --go       # call the API and record fixtures
    uv run python scripts/record_agent_scenarios.py --go --only heartbleed

Each scenario writes <fixtures>/<scenario>/{alert,result}.json plus investigator/ and critic/.
The alert is saved because its ids and timestamps are part of the first prompt: replay must send
byte-identical requests. Re-running with --go deletes and re-records the scenario directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from secops.agent.cli import _build, alert_from_event, make_llms
from secops.agent.graph import AgentDeps, run_investigation
from secops.agent.replay import recording_registry
from secops.agent.settings import get_agent_settings
from secops.config import get_settings
from secops.db.session import make_engine
from secops.schemas.alert import Alert

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "llm"

# Test-split events chosen on 2026-10-04 (see docs/agent.md for how).
SCENARIOS: dict[str, int] = {
    "ftp_bruteforce": 1110604,  # Tuesday FTP-Patator, 172.16.0.1 -> 192.168.10.50:21
    "internal_portscan": 3083227,  # Thursday scan from the infiltrated Vista host 192.168.10.8
    "benign_high_score": 357329,  # Monday (benign-only day) flow scored above threshold
    "heartbleed": 2251110,  # Wednesday Heartbleed, 172.16.0.1 -> 192.168.10.51:444
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="actually call the API and record")
    ap.add_argument("--only", nargs="*", default=None, help="subset of scenario names")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument(
        "--capture-tools",
        action="store_true",
        help="replay the recorded LLM fixtures over the real tools and save the tool outputs "
        "(no API calls); proves the recorded run is deterministic",
    )
    args = ap.parse_args()
    names = args.only or list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        print(f"unknown scenarios: {unknown}", file=sys.stderr)
        return 2

    settings = get_settings()
    agent_settings = get_agent_settings()
    engine = make_engine(settings.resolved_database_url(), read_only=True)
    registry, service = _build(settings, engine, with_detector=True)
    assert service is not None

    alerts = {n: alert_from_event(SCENARIOS[n], engine, service) for n in names}
    for n, a in alerts.items():
        p = a.prediction
        print(
            f"{n:<18} event {a.event_id}  {a.metadata.source_ip} -> {a.metadata.destination_ip}:"
            f"{a.metadata.destination_port}  p={p.attack_probability:.4f} alert={p.is_alert} "
            f"family={p.predicted_family}"
        )
    if args.capture_tools:
        for n in names:
            scenario_dir = FIXTURE_ROOT / n
            tools_dir = scenario_dir / "tools"
            if tools_dir.exists():
                shutil.rmtree(tools_dir)
            saved = json.loads((scenario_dir / "result.json").read_text())
            investigator, critic = make_llms(
                "replay", n, FIXTURE_ROOT, agent_settings.effort, agent_settings
            )
            deps = AgentDeps(
                registry=recording_registry(registry, tools_dir),
                investigator=investigator,
                critic=critic,
                tool_budget=args.budget or agent_settings.tool_budget,
            )
            alert = Alert.model_validate_json((scenario_dir / "alert.json").read_text())
            result = run_investigation(alert, deps, investigation_id=n)
            same = (
                result.report.verdict == saved["verdict"]
                and [c.tool for c in result.tool_calls] == saved["tool_calls"]
            )
            print(
                f"{n}: {len(list(tools_dir.glob('*.json')))} tool outputs captured; "
                f"replay {'matches' if same else 'DIFFERS FROM'} the recorded run"
            )
        return 0
    if not args.go:
        print("\ndry run only; pass --go to record (this calls the Anthropic API and costs money).")
        return 0

    rows = []
    total = 0.0
    for n in names:
        scenario_dir = FIXTURE_ROOT / n
        if scenario_dir.exists():
            shutil.rmtree(scenario_dir)
        scenario_dir.mkdir(parents=True)
        (scenario_dir / "alert.json").write_text(alerts[n].model_dump_json(indent=2))
        investigator, critic = make_llms(
            "record", n, FIXTURE_ROOT, agent_settings.effort, agent_settings
        )
        deps = AgentDeps(
            registry=recording_registry(registry, scenario_dir / "tools"),
            investigator=investigator,
            critic=critic,
            tool_budget=args.budget or agent_settings.tool_budget,
        )
        t0 = time.perf_counter()
        result = run_investigation(alerts[n], deps, investigation_id=n)
        wall = time.perf_counter() - t0
        u = result.usage
        summary = {
            "scenario": n,
            "event_id": SCENARIOS[n],
            "verdict": result.report.verdict,
            "severity": result.report.severity,
            "attack_family": result.report.attack_family,
            "confidence": result.report.confidence,
            "status": result.status,
            "tool_calls": [c.tool for c in result.tool_calls],
            "critic_rejections": result.iteration,
            "issues": [i.model_dump() for i in result.all_issues],
            "usage": u.model_dump(),
            "cost_usd": u.cost_usd,
            "wall_s": wall,
            "prompt_version": result.prompt_version,
            "errors": result.errors,
        }
        (scenario_dir / "result.json").write_text(json.dumps(summary, indent=2, default=str))
        (scenario_dir / "report.json").write_text(result.report.model_dump_json(indent=2))
        total += u.cost_usd
        rows.append(summary)
        print(
            f"\n{n}: {result.report.verdict} / {result.report.severity} "
            f"(conf {result.report.confidence:.2f}), {len(result.tool_calls)} tool calls, "
            f"{result.iteration} rejections, {u.calls} LLM calls, "
            f"{u.input_tokens} in / {u.output_tokens} out, ${u.cost_usd:.4f}, {wall:.0f}s"
        )
        for model, m in u.by_model.items():
            print(
                f"   {model}: {m.calls} calls, in {m.input_tokens}, out {m.output_tokens}, "
                f"cache read {m.cache_read_tokens}, cache write {m.cache_write_tokens}, "
                f"${m.cost_usd:.4f}"
            )
    print(f"\nTOTAL cost ${total:.4f} for {len(rows)} scenario(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
