"""secops dashboard: `uv run streamlit run dashboard/app.py` (needs SECOPS_API_URL and a key)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from dashboard.client import ApiClient

FIGURES = Path("docs/figures")
SEVERITY_ICON = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}


def _sev(value: str | None) -> str:
    return f"{SEVERITY_ICON.get(value or '', '')} {value or '-'}"


def get_client() -> ApiClient:
    client = st.session_state.get("client")
    if client is None:
        client = ApiClient()
        st.session_state["client"] = client
    return client  # type: ignore[no-any-return]


def page_alerts(client: ApiClient) -> None:
    st.header("Alerts")
    try:
        rows = client.investigations(limit=100)
    except Exception as e:  # the API may be down; say so instead of a traceback
        st.error(f"API unreachable: {type(e).__name__}: {e}")
        return
    if not rows:
        st.info("No investigations yet. Paste an alert below or run `secops-agent investigate`.")
    else:
        table = [
            {
                "severity": _sev(r.get("severity")),
                "verdict": r.get("verdict") or r["status"],
                "status": r["status"],
                "cost (USD)": f"{r['cost_usd']:.3f}" if r.get("cost_usd") is not None else "-",
                "created": r.get("created_at") or "-",
                "investigation_id": r["investigation_id"],
                "alert_id": r["alert_id"],
            }
            for r in rows
        ]
        st.dataframe(table, width="stretch", hide_index=True)
        chosen = st.selectbox("Open investigation", [r["investigation_id"] for r in rows])
        if st.button("Open"):
            st.session_state["selected"] = chosen
            st.session_state["page"] = "Alert detail"
            st.rerun()
    st.subheader("Investigate an alert")
    raw = st.text_area(
        "Alert JSON (as returned by POST /predict → alert)", height=160, key="alert_json"
    )
    if st.button("Start investigation") and raw.strip():
        try:
            status = client.start_investigation(json.loads(raw))
            st.success(f"queued {status['investigation_id']} ({status['status']})")
        except Exception as e:
            st.error(f"could not start: {type(e).__name__}: {e}")


def _findings(report: dict[str, Any]) -> None:
    for kind, label in (
        ("observed", "Observed (cites evidence)"),
        ("model_prediction", "Model prediction"),
        ("inference", "Inference"),
    ):
        items = [f for f in report.get("findings", []) if f["kind"] == kind]
        if items:
            st.markdown(f"**{label}**")
            for f in items:
                ids = ", ".join(f.get("evidence_ids") or []) or "-"
                st.markdown(f"- {f['statement']}  \n  evidence: `{ids}`")


def page_detail(client: ApiClient) -> None:
    st.header("Alert detail")
    inv_id = st.text_input("Investigation id", value=st.session_state.get("selected", ""))
    if not inv_id:
        st.info("Pick an investigation on the Alerts page or paste an id.")
        return
    try:
        d = client.investigation(inv_id)
    except Exception as e:
        st.error(f"not found or API unreachable: {type(e).__name__}: {e}")
        return
    st.markdown(
        f"**Status:** {d['status']} · **Verdict:** {d.get('verdict') or '-'} · "
        f"**Severity:** {_sev(d.get('severity'))} · "
        f"**Cost:** ${d.get('cost_usd') or 0:.3f}"
    )
    if d.get("error"):
        st.error(f"Investigation failed: {d['error']}")
    report = d.get("report")
    if not report:
        st.warning("No report yet (queued or running).")
        return
    st.subheader("Triage summary")
    st.write(report["summary"])
    mp = report.get("model_prediction") or {}
    st.markdown(
        f"Detector: p = {mp.get('attack_probability', 0):.4f} "
        f"(threshold {mp.get('threshold', 0):.6f}), family {mp.get('predicted_family') or '-'}, "
        f"model {mp.get('model_name')} v{mp.get('model_version')}"
    )
    if report.get("uncertainties"):
        st.subheader("Uncertainties and critic notes")
        for u in report["uncertainties"]:
            st.markdown(f"- ⚠️ {u}")
    st.subheader("Investigation timeline (tool calls)")
    for step in report.get("investigation_steps", []):
        st.code(step, language=None)
    st.subheader("Findings")
    _findings(report)
    if report.get("attack_techniques") or report.get("cves"):
        st.subheader("References")
        for t in report.get("attack_techniques", []):
            st.markdown(
                f"- ATT&CK {t['technique_id']} {t.get('name', '')} "
                f"(evidence {', '.join(t['evidence_ids'])})"
            )
        for c in report.get("cves", []):
            st.markdown(f"- {c['cve_id']} (evidence {', '.join(c['evidence_ids'])})")
    if report.get("recommended_actions"):
        st.subheader("Recommended actions")
        for a in report["recommended_actions"]:
            st.markdown(
                f"- {a['action']} (evidence {', '.join(a.get('evidence_ids') or []) or '-'})"
            )
    with st.expander("Raw report JSON"):
        st.json(report)


def page_evaluation(client: ApiClient) -> None:
    st.header("Evaluation")
    try:
        runs = client.evaluation_runs()
    except Exception as e:
        st.error(f"API unreachable: {type(e).__name__}: {e}")
        return
    if not runs:
        st.info("No evaluation runs yet (`secops-eval run`).")
    else:
        st.dataframe(runs, width="stretch", hide_index=True)
    figures = sorted(FIGURES.glob("*.png")) if FIGURES.exists() else []
    if figures:
        st.subheader("Figures (generated from the run files by scripts/make_figures.py)")
        for fig in figures:
            st.image(str(fig), caption=fig.stem)
    if len(runs) >= 2:
        st.subheader("Latest comparison")
        try:
            from secops.evaluation.compare import compare_runs, render_comparison
            from secops.evaluation.runner import RunRecord

            ordered = sorted(runs, key=lambda r: r.get("finished_at") or "")
            base = client.run_file(ordered[-2]["run_id"])
            cand = client.run_file(ordered[-1]["run_id"])
            if base and cand:
                rep = compare_runs(RunRecord.model_validate(base), RunRecord.model_validate(cand))
                st.markdown(render_comparison(rep))
        except Exception as e:
            st.warning(f"comparison unavailable: {type(e).__name__}: {e}")


def main() -> None:
    st.set_page_config(page_title="secops", layout="wide")
    st.title("Agentic Security Operations")
    pages = {"Alerts": page_alerts, "Alert detail": page_detail, "Evaluation": page_evaluation}
    default = st.session_state.get("page", "Alerts")
    choice = st.sidebar.radio("Page", list(pages), index=list(pages).index(default))
    st.session_state["page"] = choice
    client = get_client()
    try:
        h = client.health()
        st.sidebar.caption(f"API {client.base_url}: {h.get('status')}")
    except Exception:
        st.sidebar.caption(f"API {client.base_url}: unreachable")
    pages[choice](client)


main()  # streamlit executes this file as a script
