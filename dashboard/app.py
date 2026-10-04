# ruff: noqa: E501  (UI prose and Graphviz labels are long by nature)
"""secops dashboard: `./scripts/dashboard.sh` (or `uv run streamlit run dashboard/app.py`).

Works fully offline from the files in the repository (evaluation runs, recorded scenarios,
figures, README tables). With the API running (`uv run secops-api`) the Investigations pages
also show live investigations and let you start one."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # `streamlit run dashboard/app.py`

from dashboard import data  # noqa: E402
from dashboard.client import ApiClient  # noqa: E402

SEVERITY_ICON = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
VERDICT_ICON = {"true_positive": "🚨", "false_positive": "✅", "needs_human_review": "🧑‍💻"}
PIPELINE = """
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f3f4f6", fontname="Helvetica", fontsize=11];
  flows [label="network flows\\n(CIC-IDS-2017, 1.7 M)"];
  det [label="LightGBM detector\\nPR-AUC 0.9999 in-distribution\\n6 % recall on an unseen botnet", fillcolor="#e0f2fe"];
  api [label="FastAPI\\n/predict  /model  /health\\nAPI keys + rate limit"];
  alert [label="Alert\\nmetadata + probability\\n+ family + SHAP", fillcolor="#fef3c7"];
  agent [label="LangGraph agent\\nplan → investigate → critic → finalize\\nClaude Opus 5.5", fillcolor="#ede9fe"];
  tools [label="7 read-only tools\\nevent store, assets, threat intel,\\nNVD, ATT&CK, the detector"];
  report [label="TriageReport\\nverdict, severity, findings\\nciting evidence ids", fillcolor="#dcfce7"];
  eval [label="Evaluation\\n38-case golden set\\nvs rule-based baseline\\nregression gate", fillcolor="#fee2e2"];
  flows -> det -> api -> alert -> agent -> report; agent -> tools [dir=both]; report -> eval;
}
"""


def _sev(value: str | None) -> str:
    return f"{SEVERITY_ICON.get(value or '', '')} {value or '-'}"


def _verdict(value: str | None) -> str:
    return f"{VERDICT_ICON.get(value or '', '')} {value or '-'}"


def get_client() -> ApiClient:
    client = st.session_state.get("client")
    if client is None:
        client = ApiClient()
        st.session_state["client"] = client
    return client  # type: ignore[no-any-return]


def api_status(client: ApiClient) -> bool:
    try:
        return client.health().get("status") in ("ok", "degraded")
    except Exception:
        return False


# ---------------------------------------------------------------- pages
def page_overview(client: ApiClient, api_up: bool) -> None:
    st.header("What this project is")
    st.markdown(
        """
A **defensive security platform** that turns raw network traffic into triage decisions an
analyst can audit:

1. **Detect.** A LightGBM model scores every network flow (82 features) and raises an
   **alert** when the probability of an attack crosses an operating threshold chosen on a
   false-positive budget. The model is excellent on traffic it has seen and nearly blind on
   attacks it has not (6 % recall on an unseen botnet): that gap is the reason for step 2.
2. **Investigate.** A LangGraph **agent** (Claude Opus 5.5) receives the alert and asks the
   questions an analyst would, through seven **read-only tools**: what else did this source
   do, what is the target and how critical is it, is the address known, does a CVE or an
   ATT&CK technique match, do the neighbouring flows also look like attacks. It may only state
   facts that cite **evidence** the tools returned; a **critic** rejects anything else.
3. **Decide.** The result is a **triage report**: verdict (true positive, false positive, or
   needs a human), severity from a fixed rubric, findings with evidence ids, references,
   recommended actions, uncertainties, and the cost of the investigation.
4. **Measure.** A **golden set** of 38 alerts with known ground truth scores the agent
   against a **rule-based investigator** that uses the same tools without a model, and a
   **regression gate** fails any change that makes it worse.
"""
    )
    st.graphviz_chart(PIPELINE, width="stretch")
    rows = data.headline_rows()
    if rows:
        st.subheader("The headline numbers")
        st.dataframe(
            [{k: v for k, v in r.items() if k != "run_id"} for r in rows],
            width="stretch",
            hide_index=True,
        )
        st.markdown(
            """
**How to read it.** *Verdict accuracy* is the share of the 38 alerts where the final verdict
matched the ground truth. *Composite* weighs verdict, attack family, severity, evidence recall
and grounding into one score (the regression gate watches it). *Evidence recall* is whether the
investigation surfaced the facts a correct one must surface (for a port scan: the source touched
50+ ports). *Grounding* is the share of observed findings whose cited evidence really exists.
*Unsupported refs* counts ATT&CK or CVE ids that no tool returned (must be 0).

The story the three rows tell: with its model critic the agent lost to the rules on verdicts;
the evaluation measured the critic as the cause; without it the agent beats the rules on every
family of metric at about $0.12 per alert.
"""
        )
    st.subheader("How to use this dashboard")
    st.markdown(
        """
| Page | What it shows | Needs the API? |
|---|---|---|
| **Detector** | the machine-learning results from Phase 1 (which model, how good, where it fails) | no |
| **Agent results** | agent vs rule-based baseline, every metric explained, every golden case with the verdict each configuration gave | no |
| **Investigations** | the recorded investigations (38 golden cases × configurations, plus the four Phase 4 scenarios); live ones when the API runs | no (live list: yes) |
| **Investigation detail** | one triage report end to end: the alert, the tool calls, the findings with their evidence, references, uncertainties, actions, cost | no |
| **Evaluation runs** | every evaluation run file and the regression-gate comparison | no |
"""
    )
    if api_up:
        st.success(f"API reachable at {client.base_url}: you can also start live investigations.")
    else:
        st.info(
            "The API is not running, so everything shown comes from the recorded files in the "
            "repository. To start live investigations, run `uv run secops-api` in another "
            "terminal (with `SECOPS_API_KEYS` set) and `SECOPS_API_KEY` for this dashboard."
        )


def page_detector(client: ApiClient, api_up: bool) -> None:
    st.header("Detector (Phase 1)")
    st.markdown(
        """
Supervised classification of network **flows** (one row per connection, 82 statistical
features: packet sizes, timings, flag counts) from CIC-IDS-2017 in its corrected version.
Identifiers (IPs, ports) are never features. Three candidate models were trained with MLflow;
the split is chronological so the model is tested on later traffic than it was trained on, and
the operating threshold is chosen on validation data for a false-positive budget of 1 %.
"""
    )
    tables = data.readme_tables("## 5. ML pipeline", "## 6. Agent workflow")
    if len(tables) >= 1:
        st.subheader("Binary detector, test split")
        st.dataframe(data.table_to_rows(tables[0]), width="stretch", hide_index=True)
        st.caption(
            "PR-AUC: area under the precision-recall curve on the attack class (the headline "
            "metric for an imbalanced problem; accuracy is not reported on purpose). Recall / "
            "FPR / precision are at the chosen threshold. Source: README section 5, MLflow run "
            "ids in docs/evaluation.md."
        )
    fig = data.figure("detector_pr_auc")
    if fig:
        st.image(str(fig), caption="Test PR-AUC per training run (from docs/evaluation.md)")
    if len(tables) >= 2:
        st.subheader("Held-out day: train Monday–Thursday, test on Friday")
        st.dataframe(data.table_to_rows(tables[1]), width="stretch", hide_index=True)
        st.markdown(
            """
**This is the honest number.** Friday contains attack families the model never saw (Botnet,
DDoS). The in-distribution 0.9999 does not transfer: the unseen botnet is almost entirely
missed. A flow score alone is not a triage decision, which is why the rest of the platform
exists.
"""
        )
    answer = data.interview_answer("Why does the agent need tools at all")
    if answer:
        with st.expander("Why does the agent need tools at all, given a 0.9999 PR-AUC detector?"):
            st.markdown(answer)


def page_results(client: ApiClient, api_up: bool) -> None:
    st.header("Agent results (Phase 5)")
    st.markdown(
        """
**The setup.** 38 alerts from the test split with known ground truth (4 attacks per family,
the 6 benign flows the detector scores highest, 4 attacks with an injected "ignore all previous
instructions" in the tool outputs). Each is investigated by three configurations and scored on
the same rules. Expectations are derived from the labels and the severity rubric, never written
by hand, and the agent never sees them.
"""
    )
    rows = data.headline_rows()
    if not rows:
        st.info("No run files under evaluation/runs yet (`secops-eval run`).")
        return
    st.dataframe(
        [{k: v for k, v in r.items() if k != "run_id"} for r in rows],
        width="stretch",
        hide_index=True,
    )
    with st.expander("What each metric means"):
        st.markdown(
            """
- **Verdict accuracy**: final verdict equals the expected one (`true_positive` for attacks,
  `false_positive` for the benign high scorers).
- **Composite**: 0.30 verdict + 0.20 attack family + 0.15 severity within one level +
  0.20 evidence recall + 0.15 grounding. The regression gate fails a candidate whose composite
  drops more than 0.02 against the accepted run.
- **Evidence recall**: the expected evidence predicate per family was satisfied by some tool
  payload (for a brute force: at least 50 flows between the pair in the window).
- **Grounding**: observed findings whose cited evidence ids all exist. **Unsupported refs**:
  ATT&CK/CVE ids the report cites that no lookup tool returned (gate: must be 0).
- **Latency p50**: median wall-clock per investigation. **Cost**: tokens priced per model from
  the response usage, accumulated per investigation.
"""
        )
    cols = st.columns(2)
    for i, name in enumerate(
        ("agent_vs_baseline", "verdict_by_kind", "cost_latency", "reliability")
    ):
        fig = data.figure(name)
        if fig:
            cols[i % 2].image(str(fig), caption=name.replace("_", " "))
    st.markdown(
        """
**Reading.** The rule-based baseline is strong on attacks because its true-positive rule *is*
the golden-set evidence predicate, and weak on benign false positives. The agent with the
Sonnet critic surfaced more evidence than the rules but escalated too much to human review:
11 of its 13 wrong verdicts were caused by two critic rejections of findings whose numbers the
evidence did support. The 12-case ablation isolated the critic, the full-set run confirmed it,
and the rules-only configuration is now the default. Its three remaining errors are benign
DoS-shaped bursts to internal hosts, where the detector itself is wrong.
"""
    )
    st.subheader("Every golden case, with the verdict each configuration gave")
    st.caption("✅ matches the expected verdict, ❌ does not. Detector p is the alert probability.")
    st.dataframe(data.per_case_rows([r["run_id"] for r in rows]), width="stretch", hide_index=True)
    for q in (
        "Why a rule-based baseline?",
        "How do you evaluate an LLM agent without fooling yourself?",
        "What did the Phase 5 numbers show?",
    ):
        answer = data.interview_answer(q.split("?")[0])
        if answer:
            with st.expander(q):
                st.markdown(answer)


def _investigation_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": it["investigation_id"],
            "source": it["source"],
            "kind": it["kind"],
            "ground truth": it["ground_truth"],
            "verdict": _verdict(it["verdict"]),
            "expected": it["expected_verdict"],
            "severity": _sev(it["severity"]),
            "family": it.get("family") or "-",
            "confidence": it["confidence"],
            "cost (USD)": round(it["cost_usd"], 3),
            "latency (s)": it["latency_s"],
        }
        for it in items
    ]


def recorded_investigations() -> list[dict[str, Any]]:
    cached = st.session_state.get("recorded")
    if cached is None:
        cached = data.scenario_investigations() + data.investigations_from_runs(data.DEFAULT_RUN)
        st.session_state["recorded"] = cached
    return cached  # type: ignore[no-any-return]


def page_investigations(client: ApiClient, api_up: bool) -> None:
    st.header("Investigations")
    st.markdown(
        """
Each row is one alert investigated by the agent: the four **Phase 4 scenarios** (recorded live
and replayable in CI) and the 38 **golden cases** of the default configuration. The ground
truth column is what the dataset says the flow was; the agent never sees it. Pick an id and
open it on the **Investigation detail** page to see the tool calls, the evidence and the
reasoning.
"""
    )
    items = recorded_investigations()
    st.dataframe(_investigation_rows(items), width="stretch", hide_index=True)
    chosen = st.selectbox("Open", [it["investigation_id"] for it in items], key="pick_recorded")
    if st.button("Open investigation"):
        st.session_state["selected"] = chosen
        st.session_state["page"] = "Investigation detail"
        st.rerun()

    st.subheader("Live investigations (API)")
    if not api_up:
        st.info(
            "The API is not running. Start it with `uv run secops-api` (set `SECOPS_API_KEYS`) "
            "and give this dashboard `SECOPS_API_KEY` to list live investigations and start new "
            "ones; a live investigation costs about $0.12."
        )
        return
    try:
        live = client.investigations(limit=100)
    except Exception as e:
        st.error(f"API error: {type(e).__name__}: {e}")
        return
    if not live:
        st.caption("No live investigations stored yet.")
    else:
        st.dataframe(
            [
                {
                    "id": r["investigation_id"],
                    "status": r["status"],
                    "verdict": _verdict(r.get("verdict")),
                    "severity": _sev(r.get("severity")),
                    "cost (USD)": round(r["cost_usd"], 3) if r.get("cost_usd") is not None else "-",
                    "created": r.get("created_at") or "-",
                }
                for r in live
            ],
            width="stretch",
            hide_index=True,
        )
        live_pick = st.selectbox(
            "Open live", [r["investigation_id"] for r in live], key="pick_live"
        )
        if st.button("Open live investigation"):
            st.session_state["selected"] = f"live:{live_pick}"
            st.session_state["page"] = "Investigation detail"
            st.rerun()
    raw = st.text_area(
        "Start a new investigation: paste an Alert JSON (for example tests/fixtures/llm/ftp_bruteforce/alert.json)",
        height=140,
        key="alert_json",
    )
    if st.button("Start investigation") and raw.strip():
        try:
            status = client.start_investigation(json.loads(raw))
            st.success(
                f"queued {status['investigation_id']} ({status['status']}); reload in a minute"
            )
        except Exception as e:
            st.error(f"could not start: {type(e).__name__}: {e}")


def _find_recorded(inv_id: str) -> dict[str, Any] | None:
    for it in recorded_investigations():
        if it["investigation_id"] == inv_id:
            return it
    return None


def _render_report(report: dict[str, Any], meta: dict[str, Any] | None = None) -> None:
    mp = report.get("model_prediction") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verdict", report["verdict"])
    c2.metric("Severity", report["severity"])
    c3.metric("Confidence", f"{report['confidence']:.2f}")
    c4.metric("Attack family", report.get("attack_family") or "-")
    if meta:
        st.caption(
            f"Source: {meta['source']} · ground truth: {meta['ground_truth']} · expected verdict: "
            f"{meta['expected_verdict']} · cost ${meta['cost_usd']:.3f} · {meta['latency_s']} s"
        )
    st.subheader("1. The alert the detector raised")
    st.markdown(
        f"Attack probability **{mp.get('attack_probability', 0):.4f}** against a threshold of "
        f"{mp.get('threshold', 0):.6f}; the family classifier said **{mp.get('predicted_family') or '-'}**; "
        f"model {mp.get('model_name')} v{mp.get('model_version')}."
    )
    st.subheader("2. What the agent did (tool calls, in order)")
    st.caption("Each call produced an evidence id (E1, E2, …) that the findings below cite.")
    for step in report.get("investigation_steps", []):
        st.code(step, language=None)
    st.subheader("3. What the agent concluded")
    st.write(report["summary"])
    st.subheader("4. Findings, with the evidence they cite")
    for kind, label, hint in (
        ("observed", "Observed", "facts taken from tool results; every one cites evidence ids"),
        ("model_prediction", "Model prediction", "what the detector said"),
        ("inference", "Inference", "the agent's interpretation, kept apart from facts"),
    ):
        items = [f for f in report.get("findings", []) if f["kind"] == kind]
        if items:
            st.markdown(f"**{label}** — *{hint}*")
            for f in items:
                ids = ", ".join(f.get("evidence_ids") or []) or "-"
                st.markdown(f"- {f['statement']}  \n  evidence: `{ids}`")
    if report.get("attack_techniques") or report.get("cves"):
        st.subheader("5. References (only ids a lookup tool returned)")
        for t in report.get("attack_techniques", []):
            st.markdown(
                f"- ATT&CK **{t['technique_id']}** {t.get('name', '')} (evidence {', '.join(t['evidence_ids'])})"
            )
        for c in report.get("cves", []):
            st.markdown(f"- **{c['cve_id']}** (evidence {', '.join(c['evidence_ids'])})")
    if report.get("recommended_actions"):
        st.subheader("6. Recommended actions")
        for a in report["recommended_actions"]:
            st.markdown(
                f"- {a['action']} (evidence {', '.join(a.get('evidence_ids') or []) or '-'})"
            )
    if report.get("uncertainties"):
        st.subheader("7. Uncertainties and critic notes")
        st.caption("What the agent could not establish, tool failures, and critic objections.")
        for u in report["uncertainties"]:
            st.markdown(f"- ⚠️ {u}")
    with st.expander("Raw report JSON"):
        st.json(report)


def page_detail(client: ApiClient, api_up: bool) -> None:
    st.header("Investigation detail")
    st.markdown(
        "One triage report, end to end. Severity never comes from the model: it is a fixed "
        "rubric over the attack family, the asset's criticality and a success indicator computed "
        "from evidence. A verdict of *needs_human_review* means the agent escalated with its "
        "evidence attached rather than guessing."
    )
    items = recorded_investigations()
    ids = [it["investigation_id"] for it in items]
    default = st.session_state.get("selected") or (ids[0] if ids else "")
    inv_id = (
        st.selectbox(
            "Investigation (recorded) — or paste a live id below",
            ids,
            index=ids.index(default) if default in ids else 0,
            key="detail_pick",
        )
        if ids
        else ""
    )
    live_id = st.text_input(
        "Live investigation id (needs the API)",
        value=default[5:] if default.startswith("live:") else "",
    )
    if live_id:
        if not api_up:
            st.info("The API is not running; start it to read live investigations.")
            return
        try:
            d = client.investigation(live_id)
        except Exception as e:
            st.error(f"not found or API error: {type(e).__name__}: {e}")
            return
        st.markdown(f"**Status:** {d['status']}")
        if d.get("error"):
            st.error(f"Investigation failed: {d['error']}")
        if d.get("report"):
            _render_report(d["report"])
        else:
            st.warning("No report yet (queued or running).")
        return
    it = _find_recorded(inv_id)
    if it is None:
        st.info("Pick an investigation on the Investigations page.")
        return
    _render_report(it["report"], it)


def page_evaluation(client: ApiClient, api_up: bool) -> None:
    st.header("Evaluation runs")
    st.markdown(
        """
Every evaluation run is a JSON file under `evaluation/runs/` with its configuration, per-case
scores and reports, and aggregated metrics. `evaluation/baselines/latest.json` is the accepted
run; CI replays five recorded cases at zero cost and compares. The gates: composite drop
≤ 0.02, grounding drop ≤ 0.02, cost rise ≤ 25 %, zero unsupported references, identical case
set and golden-set version.
"""
    )
    runs = data.list_runs()
    if not runs:
        st.info("No evaluation runs yet (`secops-eval run`).")
        return
    st.dataframe(
        [
            {
                "run": r["config"]["run_id"],
                "investigator": r["config"]["investigator"],
                "critic": "rules + model"
                if r["config"].get("llm_critic", True) and r["config"]["investigator"] == "agent"
                else ("rules only" if r["config"]["investigator"] == "agent" else "-"),
                "cases": r["metrics"]["cases"],
                "repeats": r["metrics"]["repeats"],
                "verdict accuracy": round(data.mean(r["metrics"], "verdict_accuracy"), 3),
                "composite": round(r["metrics"]["composite"], 3),
                "grounding": round(data.mean(r["metrics"], "grounding_rate"), 3),
                "cost total (USD)": round(r["metrics"]["cost_total_usd"], 3),
                "finished": (r.get("finished_at") or "")[:19],
            }
            for r in runs
        ],
        width="stretch",
        hide_index=True,
    )
    st.subheader("Compare two runs with the regression gate")
    ids = [r["config"]["run_id"] for r in runs]
    c1, c2 = st.columns(2)
    base = c1.selectbox(
        "Baseline",
        ids,
        index=ids.index("baseline-rule-based") if "baseline-rule-based" in ids else 0,
    )
    cand = c2.selectbox(
        "Candidate", ids, index=ids.index(data.DEFAULT_RUN) if data.DEFAULT_RUN in ids else 0
    )
    try:
        from secops.evaluation.compare import compare_runs, render_comparison
        from secops.evaluation.runner import RunRecord

        b = next(r for r in runs if r["config"]["run_id"] == base)
        c = next(r for r in runs if r["config"]["run_id"] == cand)
        rep = compare_runs(RunRecord.model_validate(b), RunRecord.model_validate(c))
        st.markdown(render_comparison(rep))
    except Exception as e:
        st.warning(f"comparison unavailable: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- main
PAGES = {
    "Overview": page_overview,
    "Detector": page_detector,
    "Agent results": page_results,
    "Investigations": page_investigations,
    "Investigation detail": page_detail,
    "Evaluation runs": page_evaluation,
}


def main() -> None:
    st.set_page_config(page_title="Agentic Security Operations", layout="wide")
    st.title("Agentic Security Operations Platform")
    default = st.session_state.get("page", "Overview")
    choice = st.sidebar.radio("Page", list(PAGES), index=list(PAGES).index(default))
    st.session_state["page"] = choice
    client = get_client()
    api_up = api_status(client)
    st.sidebar.caption(
        f"API {client.base_url}: {'reachable' if api_up else 'not running (offline mode)'}"
    )
    st.sidebar.markdown(
        "Offline mode shows the recorded investigations and results. "
        "`uv run secops-api` enables live investigations."
    )
    st.sidebar.markdown("Docs: `docs/project-guide.md` · `docs/evaluation.md` · `docs/agent.md`")
    PAGES[choice](client, api_up)


main()  # streamlit executes this file as a script
