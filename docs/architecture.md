# Architecture

The diagram is the one in the README (kept here as the source of truth; GitHub renders Mermaid
inline, so no exported image is needed). Each box is a module or service that exists in the
repository; the arrows are the real call paths.

```mermaid
flowchart LR
  subgraph data [Data and ML]
    D[CIC-IDS-2017 DistriNet] --> P[clean / dedup / chrono split]
    P --> T[LightGBM / XGBoost / LogReg<br/>MLflow runs + registry]
    T --> B[model bundle<br/>estimator + spec + threshold + SHAP]
  end
  subgraph api [FastAPI service]
    B --> API[/predict  /model  /health<br/>API keys + per-key rate limit/]
    API --> AL[Alert: metadata + probability + family + SHAP]
  end
  subgraph agent [LangGraph investigation]
    AL --> PL[plan] --> IN[investigate<br/>Opus 5.5, tool budget 12]
    IN --> CR[critic<br/>rules + Sonnet 5.5] -->|rejected ≤2| IN
    CR --> FI[finalize<br/>severity rubric] --> TR[TriageReport]
  end
  subgraph tools [Read-only tools]
    IN --> ES[(event store<br/>1.7 M flows, SQLite/PostgreSQL)]
    IN --> SE[assets / threat intel seeds]
    IN --> NV[NVD CVE cache]
    IN --> AT[ATT&CK v19.2 index]
    IN --> DT[predict_attack]
  end
  TR --> DB[(investigations<br/>tool_calls)]
  TR --> LF[Langfuse traces]
  DB --> UI[Streamlit dashboard]
  subgraph eval [Evaluation]
    G[golden set v1<br/>38 test-split alerts] --> R[runner: agent / rule-based<br/>k repeats, record / replay]
    R --> C[compare: composite, grounding,<br/>cost gates]
  end
```

## Reading the diagram

- **Data and ML** (`src/secops/data`, `src/secops/detection`): the corrected CIC-IDS-2017 flows
  are cleaned, deduplicated and split chronologically; LightGBM, XGBoost and logistic regression
  are trained under MLflow; the champion is exported as a self-contained bundle.
- **FastAPI service** (`src/secops/api`): serves predictions from the bundle, authenticates with
  API keys, rate-limits per key, and turns above-threshold predictions into `Alert` objects.
- **LangGraph investigation** (`src/secops/agent`): plan → investigate → critic → finalize with a
  tool budget, at most two critic rejections, a deterministic severity rubric and record/replay
  of every model call.
- **Read-only tools** (`src/secops/tools`): typed, bounded, label-free; the detector is one of
  them.
- **Persistence and observability** (`src/secops/db`, `src/secops/observability`): investigations
  and tool calls in PostgreSQL or SQLite through Alembic migrations; one trace per investigation
  when Langfuse keys are configured.
- **Evaluation** (`src/secops/evaluation`): the golden set, the rule-based baseline, the runner
  with repeats and replay, and the regression gate.
- **Dashboard** (`dashboard/`): Streamlit over the API, three pages.
