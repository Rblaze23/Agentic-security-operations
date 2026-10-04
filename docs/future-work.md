# Future work

Each item names the measurement that would justify it; none is started.

| Area | What | Why it is not done yet |
|---|---|---|
| Model critic | Re-prompt the Sonnet critic (or replace the per-finding judgement with a structured "claims vs evidence fields" check) and measure it against the rules-only default on the full set | The full 38-case run showed the current prompt is a net negative (0.658 vs 0.921 verdict accuracy, equal grounding); the default is rules-only until a critic prompt measures better |
| Deployed agent | Cloud SQL for investigations and a Cloud Storage copy of the event store so the agent endpoints can run on Cloud Run | A cost decision for an owner; the public deployment serves the detector only |
| Multi-replica safety | Shared rate-limit store (Redis) and an investigation queue | Not needed for one replica; both are documented limitations |
| Tool payload rendering | Compact per-flow lines for `predict_attack` and `search_events` instead of a 4,000-character JSON cut | Two recorded reports mention "output was cut off" in their uncertainties |
| Event store at scale | Single-pass aggregate query for `get_related_events` on DoS-burst anchors (p95 1.1 s), PostgreSQL in production | Measured but not a bottleneck for 38-case evaluations |
| Retraining and drift | MLflow alias swap for promotion is in place; add feature-distribution PSI on served predictions (`model_predictions` table exists) and a scheduled retrain on new labelled data | No new data arrives in a static dataset |
| Observability | Verify the Langfuse tracer against a Langfuse instance; add cost dashboards per prompt version | No Langfuse keys were available during development |
| Detector generalisation | The held-out Friday experiment shows 6 % botnet recall: training on more days, or a second detector on the Friday families, would be the next ML step | Would need a different split protocol to evaluate honestly |
| Benign false positives | The three remaining errors of the default configuration are DoS-shaped benign bursts to internal hosts; a benign-burst rule in the baseline and a comparison of the detector's own calibration there would say whether this is an agent or a detector problem | Needs more such cases than the six in v1 |
| Golden set growth | More benign false positives and the Friday families (Botnet, DDoS, PortScan with the held-out split) | Budget rule: the golden set stays small until the critic question is settled |
