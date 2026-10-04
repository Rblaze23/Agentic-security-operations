# Demo walkthrough

Two ways to see the platform work: offline in about a minute (no key, no model, no data), or
live against the Anthropic API with the full event store.

## Offline (replayed), the way CI runs it

```bash
uv run python scripts/demo.py
```

What happens, step by step:

1. Fixture bundles (a small LightGBM detector and family classifier trained on the 986-row test
   fixture) are built into a temporary directory, and the API starts on port 18010 in replay
   mode with a temporary SQLite database.
2. The recorded FTP brute-force alert (`tests/fixtures/llm/ftp_bruteforce/alert.json`: Tuesday
   flow 1110604, 172.16.0.1 → 192.168.10.50:21, detector probability 0.9999, family
   brute_force) is submitted with `POST /investigations` and comes back `202 queued`.
3. The LangGraph investigation runs in the background: the planner, five tool calls
   (`get_related_events`, `get_asset`, `enrich_ip`, `predict_attack`, `lookup_attack_technique`),
   the draft report, the critic. Every model response and every tool output is served from the
   fixtures recorded on 2026-10-04, so the run is identical every time.
4. `GET /investigations/ftp_bruteforce` returns the persisted report: verdict `true_positive`,
   severity `high` (brute force on a high-criticality web server), cost $0.1222 (the recorded
   usage), eight findings with evidence ids, ATT&CK T1110.001, four uncertainties.

The printed output is the same summary the dashboard shows on the alert detail page.

## Live

```bash
export ANTHROPIC_API_KEY=...          # or put it in .env
make compose-up                       # postgres + api with the real bundles and event store
uv run secops-agent investigate --event-id 1110604     # same alert, real model, persisted
uv run secops-agent show <investigation_id>
make dashboard                        # http://localhost:8501
```

A live investigation costs about $0.12 to $0.28 (Phase 4 and Phase 5 measurements) and takes
30 to 60 seconds; the CLI prints the cost after the run and `secops-eval` prints the estimate
before a batch.

## What to look at in two minutes

- `docs/agent.md`: the graph, the evidence rules, the measured scenario table.
- `docs/evaluation.md` Phase 5: the golden set, the agent-vs-baseline table, the gates.
- `tests/unit/agent/test_graph.py`: the behaviours that are pinned (budget exhaustion, double
  rejection, tool errors, injected instructions, refusal fallback).
