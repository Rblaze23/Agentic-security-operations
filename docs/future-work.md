# Future work

Each item names the measurement that would justify it; none is started.

| Area | What | Why it is not done yet |
|---|---|---|
| Critic precision | Re-prompt the Sonnet critic (or replace the per-finding judgement with a structured "claims vs evidence fields" check) and run the full 38-case set with and without it; consider making the rules-only critic the default | The 12-case ablation already shows 0.917 vs 0.583 verdict accuracy without vs with the model critic at equal grounding; the full-set confirmation costs about $7 and was not run inside the Phase 5 budget |
| Deployed agent | Cloud SQL for investigations and a Cloud Storage copy of the event store so the agent endpoints can run on Cloud Run | A cost decision for an owner; the public deployment serves the detector only |
| Multi-replica safety | Shared rate-limit store (Redis) and an investigation queue | Not needed for one replica; both are documented limitations |
| Tool payload rendering | Compact per-flow lines for `predict_attack` and `search_events` instead of a 4,000-character JSON cut | Two recorded reports mention "output was cut off" in their uncertainties |
| Event store at scale | Single-pass aggregate query for `get_related_events` on DoS-burst anchors (p95 1.1 s), PostgreSQL in production | Measured but not a bottleneck for 38-case evaluations |
| Retraining and drift | MLflow alias swap for promotion is in place; add feature-distribution PSI on served predictions (`model_predictions` table exists) and a scheduled retrain on new labelled data | No new data arrives in a static dataset |
| Observability | Verify the Langfuse tracer against a Langfuse instance; add cost dashboards per prompt version | No Langfuse keys were available during development |
| Detector generalisation | The held-out Friday experiment shows 6 % botnet recall: training on more days, or a second detector on the Friday families, would be the next ML step | Would need a different split protocol to evaluate honestly |
| Golden set growth | More benign false positives and the Friday families (Botnet, DDoS, PortScan with the held-out split) | Budget rule: the golden set stays small until the critic question is settled |
