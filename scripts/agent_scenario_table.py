"""Print the measured scenario table for docs/agent.md from tests/fixtures/llm/*/result.json."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "llm"
ORDER = ["ftp_bruteforce", "internal_portscan", "benign_high_score", "heartbleed"]


def main() -> None:
    rows = []
    for name in ORDER:
        p = ROOT / name / "result.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        u = r["usage"]["by_model"]
        opus = u.get("claude-opus-5-5", {})
        son = u.get("claude-sonnet-5-5", {})
        cache_share = opus.get("cache_read_tokens", 0) / max(
            1, opus.get("input_tokens", 0) + opus.get("cache_read_tokens", 0)
        )
        rows.append(
            f"| `{name}` | {r['event_id']} | {r['verdict']} | {r['severity']} | "
            f"{r['attack_family'] or '-'} | {r['confidence']:.2f} | {len(r['tool_calls'])} | "
            f"{r['critic_rejections']} | {opus.get('calls', 0)} + {son.get('calls', 0)} | "
            f"{opus.get('input_tokens', 0) + opus.get('cache_read_tokens', 0):,} | "
            f"{cache_share:.0%} | {opus.get('output_tokens', 0):,} | ${r['cost_usd']:.3f} | "
            f"{r['wall_s']:.0f} s |"
        )
    print(
        "| Scenario | Event | Verdict | Severity | Family | Conf. | Tool calls | "
        "Critic rejections | LLM calls (Opus + Sonnet) | Opus input tokens (incl. cache reads) | "
        "Cache-read share | Opus output tokens | Cost | Wall-clock |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    total = sum(
        json.loads((ROOT / n / "result.json").read_text())["cost_usd"]
        for n in ORDER
        if (ROOT / n / "result.json").exists()
    )
    print(f"\nTotal recorded cost: ${total:.3f}")


if __name__ == "__main__":
    main()
