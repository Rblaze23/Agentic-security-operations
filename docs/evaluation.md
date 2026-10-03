# Evaluation

Every number below was produced by `secops-train` on 2026-10-03 (second, final run after the
review fix pass) and lives in MLflow under `$SECOPS_DATA_DIR/mlflow` (SQLite backend, artifacts
under `mlflow/artifacts/`). Tables are pasted from `secops-train report`; the analysis numbers in
sections 2.1, 4 and 5 come from the per-run artifacts `per_label_recall.csv`,
`threshold_sweep.json` and `fp_breakdown.json`; run ids are listed in section 7 so each row can be
traced. The agent-evaluation sections arrive with Phase 5.

## 1. Protocol in one paragraph

Data: DistriNet-improved CIC-IDS-2017, deduplicated, Attempted flows relabelled benign (see
`docs/dataset.md`). Primary split: chronological inside each (day, label) group, 70/15/15. Models
are selected on the **validation** split only (early stopping, operating threshold, champion
choice) and the **test** split is scored once per configuration. Headline metric for the binary
detector is PR-AUC on the attack class; the operating point is the threshold with the highest
validation recall whose validation false-positive rate is at most 1%. Headline for the family
classifier is macro-F1. Accuracy is logged but never used for a decision.

## 2. Binary detector (`secops/detection-binary`)

Six runs: three models × {no weighting, balanced sample weights}. Test split: 257,252 flows,
42,368 attacks (16.5%).

| run_name          | model    | weighting   |   val_pr_auc |   test_pr_auc |   test_recall |   test_fpr |   test_precision |   test_f1 |   test_roc_auc |   threshold | run_id   |
|:------------------|:---------|:------------|-------------:|--------------:|--------------:|-----------:|-----------------:|----------:|---------------:|------------:|:---------|
| lightgbm_none     | lightgbm | none        |       1      |        0.9999 |        0.9993 |     0.0045 |           0.9778 |    0.9884 |         1      |      0.0002 | dcc36c43 |
| lightgbm_balanced | lightgbm | balanced    |       1      |        0.9999 |        0.9997 |     0.0087 |           0.9576 |    0.9782 |         1      |      0      | 7335febb |
| xgboost_none      | xgboost  | none        |       1      |        0.9999 |        0.9997 |     0.0076 |           0.9628 |    0.9809 |         1      |      0.0002 | a567ffb7 |
| xgboost_balanced  | xgboost  | balanced    |       1      |        0.9999 |        0.9995 |     0.0053 |           0.974  |    0.9866 |         1      |      0.0008 | 040372e7 |
| logreg_none       | logreg   | none        |       0.9945 |        0.9936 |        0.9797 |     0.007  |           0.9649 |    0.9722 |         0.9973 |      0.1769 | a77a701f |
| logreg_balanced   | logreg   | balanced    |       0.9947 |        0.993  |        0.9816 |     0.0057 |           0.9713 |    0.9764 |         0.9979 |      0.528  | 0867b1bc |

Validation PR-AUC at full precision: xgboost_balanced 0.999995, xgboost_none 0.999994,
lightgbm_none 0.999993, lightgbm_balanced 0.999993, logreg_balanced 0.994671, logreg_none 0.994478.

**Champion: `lightgbm_none` (run dcc36c43), registered as `secops-detector` version 1, alias
`champion`.** Rule, implemented in `secops.detection.selection.choose_champion` and applied by
`secops-train promote-best`: highest validation PR-AUC among finished runs; every run within
0.0005 of the best is a tie, resolved by the simpler model (logreg < lightgbm < xgboost), then by
the lower validation FPR at the operating point. Here the four tree runs tie within 0.000002, so
LightGBM wins over XGBoost, and the unweighted LightGBM wins on validation FPR (0.0043 vs 0.0077).
The command's own output: `tie within 0.0005 of best val_pr_auc (4 runs): simpler model, then
lower val_fpr`.

Champion operating point (chosen on validation): threshold 0.000242, validation FPR 0.429%,
validation recall 99.998%. On test: recall 99.93%, FPR 0.45%, precision 97.78%, 963 false positives
and 31 missed attacks out of 257,252 flows; Brier score 0.0013. Recall at fixed test FPR: 99.88% at
0.1%, 99.93% at 0.5%, 99.99% at 1%, 99.99% at 5%. The max-F1 threshold would be 0.322; the FPR rule
picks a far lower one because the score distribution is almost perfectly bimodal.

Wall-clock on 8 cores (`fit_seconds` metric): LightGBM 21–34 s, XGBoost 41–47 s, logistic
regression 74–89 s; SHAP on 20,000 validation rows (`explain_seconds`): LightGBM 15–21 s, XGBoost
4 s, logistic regression under 1 s.

### 2.1 Per-label recall on the test split (binary task, champion threshold)

| Original label | lightgbm_none (champion) | logreg_balanced |
|---|---|---|
| Botnet | 1.000 | 0.027 |
| DDoS | 1.000 | 0.998 |
| DoS GoldenEye | 1.000 | 0.908 |
| DoS Hulk | 1.000 | 0.999 |
| DoS Slowhttptest | 1.000 | 0.318 |
| DoS Slowloris | 0.998 | 0.860 |
| FTP-Patator | 1.000 | 0.998 |
| Heartbleed (2 test rows) | 1.000 | 1.000 |
| Infiltration (6 test rows) | 1.000 | 0.000 |
| Infiltration - Portscan | 1.000 | 0.865 |
| Portscan | 0.960 | 0.628 |
| SSH-Patator | 0.955 | 0.912 |
| Web Attack - Brute Force (11 test rows) | 1.000 | 1.000 |
| Web Attack - SQL Injection (2 test rows) | 1.000 | 0.000 |
| Web Attack - XSS (3 test rows) | 1.000 | 1.000 |

The aggregate PR-AUC hides this: logistic regression is a 0.993 PR-AUC model that misses almost
every Botnet, Infiltration, Slowhttptest and SQL-injection flow. The tree model's weakest labels
are the two "many identical probes" behaviours (Portscan, SSH-Patator).

### 2.2 How to read these numbers

- **Why so high.** The primary split is chronological *within* each (day, label) group: the model
  trains on the first 70% of every attack burst and is tested on the last 15% of the same burst,
  on the same testbed. Flows inside one tool's burst are near-identical. This is the "we have seen
  the start of the campaign" setting and it is easy by construction. The held-out-Friday
  experiment (section 4) is the number to quote for unseen attacks.
- **Weighting did nothing measurable** for the binary task; the operating threshold is the lever.
- **Validation is used twice** (early stopping and threshold selection); test is untouched.
- **Deduplication** removed 98.9% of Portscan rows, so port-scan metrics are computed on 1,173 test
  flows that represent what a *single* probe looks like, not a scan's volume.

### 2.3 Global SHAP importance (champion, 20,000 validation flows)

| Rank | Feature | mean abs SHAP |
|---|---|---|
| 1 | Bwd Packet Length Std | 2.693 |
| 2 | Packet Length Std | 0.666 |
| 3 | Bwd Init Win Bytes | 0.610 |
| 4 | Bwd Packet Length Mean | 0.572 |
| 5 | Total Length of Bwd Packet | 0.277 |
| 6 | Packet Length Max | 0.171 |
| 7 | Fwd IAT Min | 0.143 |
| 8 | Fwd Packet Length Max | 0.140 |
| 9 | Bwd Segment Size Avg | 0.126 |
| 10 | Packet Length Variance | 0.121 |

Response-side (backward) packet statistics dominate: attack tools elicit unusually regular replies.
Per-flow top-k contributions are produced by `top_k_contributions` and will be attached to each
prediction in Phase 2.

## 3. Attack-family classifier (`secops/detection-family`)

Attack rows only, six classes (`rare_exploit` excluded, 47 rows). Test split: 42,360 flows.

| run_name                 | model    | weighting   |   val_macro_f1 |   test_macro_f1 | test_weighted_f1 | test_log_loss | run_id   |
|:-------------------------|:---------|:------------|---------------:|----------------:|-----------------:|--------------:|:---------|
| lightgbm_family_none     | lightgbm | none        |         0.9996 |          0.9613 |           0.9989 |        0.0030 | 79e4b102 |
| xgboost_family_balanced  | xgboost  | balanced    |         0.9991 |          0.9616 |           0.9987 |        0.0081 | 9682dde2 |
| xgboost_family_none      | xgboost  | none        |         0.9946 |          0.9978 |           0.9992 |        0.0042 | 66a8b919 |
| lightgbm_family_balanced | lightgbm | balanced    |         0.9905 |          0.9708 |           0.9986 |        0.0078 | 7c851e2a |

**Champion: `lightgbm_family_none` (run 79e4b102), registered as `secops-family-classifier`
version 1, alias `champion`.** The two best validation scores (0.999552 and 0.999103) fall inside
the 0.0005 tie band, so the simpler model wins (`promote-best`: `tie within 0.0005 of best
val_macro_f1 (2 runs): simpler model, then lower val_fpr`). Its test macro-F1 (0.9613) is lower
than xgboost_family_none's (0.9978). Selecting on validation and reporting on test is the
discipline; swapping champions after looking at test would be the leak the protocol exists to
prevent. Balanced weights helped XGBoost on validation (0.9991 vs 0.9946) and hurt LightGBM
(0.9905 vs 0.9996).

Per-class on test, champion:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| botnet | 0.974 | 1.000 | 0.987 | 111 |
| brute_force | 0.999 | 0.992 | 0.996 | 1,041 |
| ddos | 1.000 | 0.9985 | 0.999 | 14,272 |
| dos | 0.9995 | 0.9997 | 1.000 | 25,747 |
| port_scan | 0.981 | 0.992 | 0.986 | 1,173 |
| web_attack (low support) | 0.667 | 1.000 | 0.800 | 16 |

Most of the macro-F1 gap is `web_attack`: 7 DoS flows and 1 brute-force flow were predicted as web
attack, and against 16 true web-attack rows that is a precision of 0.667. Confusion matrix (rows
actual, columns predicted, order botnet / brute_force / ddos / dos / port_scan / web_attack):

```
botnet       [  111,     0,     0,     0,    0,  0]
brute_force  [    3,  1033,     0,     3,    1,  1]
ddos         [    0,     0, 14250,     1,   21,  0]
dos          [    0,     1,     0, 25738,    1,  7]
port_scan    [    0,     0,     0,     9, 1164,  0]
web_attack   [    0,     0,     0,     0,    0, 16]
```

Weighted F1 (0.9989) says the classifier is right on almost every flow; macro-F1 says the smallest
class is where the errors concentrate. Both are true, and the 104-row `web_attack` class is flagged
low-support everywhere it appears.

## 4. Held-out Friday (`secops/detection-heldout`)

Champion configuration (LightGBM, no weighting) trained on Monday–Thursday only, threshold
**reused unchanged** from the chrono champion (0.000242), tested on all 346,530 Friday flows.
Botnet and DDoS never appear in training.

| run_name                     | model    |   val_pr_auc |   test_pr_auc |   test_recall |   test_fpr |   test_precision |   test_f1 |   test_roc_auc |   threshold | run_id   |
|:-----------------------------|:---------|-------------:|--------------:|--------------:|-----------:|-----------------:|----------:|---------------:|------------:|:---------|
| lightgbm_none_heldout_friday | lightgbm |       0.9999 |        0.9991 |        0.9928 |     0.0167 |           0.9589 |    0.9755 |         0.9997 |      0.0002 | a174088a |

Per-label recall on Friday at the reused threshold: **Botnet 6.0%**, DDoS 100%, Portscan 99.2%
(4,154 false positives, 706 misses). Brier score 0.206 (versus 0.0013 in-distribution): the
probabilities are badly calibrated on unseen attacks.

What the same model does at other thresholds, from the run's `threshold_sweep.json` (reference
only; thresholds are never tuned on test):

| Threshold | Recall | FPR | Precision | Botnet | DDoS | Portscan |
|---|---|---|---|---|---|---|
| 0.000242 (reused, deployed) | 99.3% | 1.67% | 95.9% | 6.0% | 100% | 99.2% |
| 0.001 | 99.1% | 0.56% | 98.6% | 0% | 100% | 92.3% |
| 0.01 | 98.5% | 0.07% | 99.8% | 0% | 100% | 57.9% |
| 0.05 | 96.7% | 0.01% | 99.97% | 0% | 98.6% | 28.5% |
| 0.1 | 87.5% | 0.01% | 99.98% | 0% | 89.3% | 24.1% |
| 0.25 | 7.1% | 0.00% | 99.9% | 0% | 7.0% | 13.6% |
| 0.5 (conventional) | 1.3% | 0.00% | 99.8% | 0% | 1.2% | 8.7% |

So every Botnet flow scores below 0.001, 98.8% of DDoS flows score below 0.5, and 71.5% of
Portscan flows score below 0.05.

Reading: the in-distribution 0.9999 PR-AUC does not transfer. An unseen command-and-control
pattern is almost entirely missed; the DDoS flood is caught only because the deployed threshold is
tiny, and the FPR budget is overshot (1.67% against a 1% target) because the threshold was fixed
on a different week. The Friday PR-AUC of 0.9991 is still high because DDoS (95,144 flows) dominates
the positives; the per-label table is the number that matters. This is the measured case for the
rest of the platform: a flow score is a signal, not a triage decision, until it is combined with
burst counts, asset context and threat intelligence.

## 5. Ablations (`secops/detection-ablations`)

Champion configuration with exactly one variable changed.

| run_name                     | change | val_pr_auc | test_pr_auc | test_recall | test_fpr | test_precision | threshold | FN | FP | run_id |
|:-----------------------------|:-------|-----------:|------------:|------------:|---------:|---------------:|----------:|---:|---:|:-------|
| lightgbm_none (champion)     | —      | 0.999993 | 0.999931 | 99.93% | 0.45% | 97.78% | 0.00024 | 31 | 963 | dcc36c43 |
| lightgbm_none_withport       | `Dst Port` added as a feature | 0.999994 | 0.999971 | 99.99% | 0.46% | 97.74% | 0.00004 | 3 | 979 | d10efe46 |
| lightgbm_none_attempted_drop | Attempted flows dropped instead of relabelled benign | 0.999998 | 0.999962 | 99.82% | 0.012% | 99.94% | 0.0016 | 75 | 25 | 0cc2305b |

**Port.** With the destination port the model misses 3 attack flows instead of 31 and every
per-label recall except Portscan reaches 1.0. In this testbed attacks target five ports, so the
port is a shortcut that would not survive a different network; the gain confirms the decision to
exclude it and to let the Phase 3 tools, not the model, reason about ports.

**Attempted policy.** PR-AUC is unchanged within noise (0.999962 vs 0.999931). The operating-point
differences (25 vs 963 false positives, 75 vs 31 misses) come from where the FPR rule placed the
threshold on a near-separable validation set (0.0016 vs 0.00024), not from the policy itself: the
champion's `fp_breakdown.json` shows only 7 of its 963 false positives are relabelled Attempted
flows (all `Botnet - Attempted`). The authors' recommended policy (relabel benign) stays the
default.

### 5.1 Where the champion's false positives come from (`fp_breakdown.json`)

| Day of the 963 test false positives | Count |
|---|---|
| Friday (includes the 7 `Botnet - Attempted` flows) | 652 |
| Monday | 86 |
| Thursday | 80 |
| Tuesday | 73 |
| Wednesday | 72 |

By raw label: 956 BENIGN, 7 `Botnet - Attempted`. Friday's benign background traffic is where the
detector is least sure of itself, which matches the held-out result: Friday looks different.

## 6. Reproducing

```bash
export SECOPS_DATA_DIR=$HOME/data/secops
uv run secops-data verify && make data-build && make train-all
uv run secops-train report --experiment secops/detection-binary
uv run secops-train promote-best --experiment secops/detection-binary --model-name secops-detector
```

`make train-all` was run twice on 2026-10-03; the second run (after the review fix pass, with the
artifact root under the data directory and the promotion rule in code) produced identical binary,
held-out and ablation metrics to four decimals and the same two champions; the LightGBM family run
moved by 0.0004 validation macro-F1 (LightGBM is not bit-reproducible across thread schedules).

## 7. Run ids

| run_name | experiment | run_id |
|---|---|---|
| lightgbm_none (champion) | secops/detection-binary | dcc36c43a75749eab6c77e55e79cf837 |
| lightgbm_balanced | secops/detection-binary | 7335febb |
| xgboost_balanced | secops/detection-binary | 040372e7 |
| xgboost_none | secops/detection-binary | a567ffb7 |
| logreg_balanced | secops/detection-binary | 0867b1bc |
| logreg_none | secops/detection-binary | a77a701f |
| lightgbm_family_none (champion) | secops/detection-family | 79e4b102f4c14c57951e91d61496beac |
| lightgbm_family_balanced | secops/detection-family | 7c851e2a |
| xgboost_family_balanced | secops/detection-family | 9682dde2 |
| xgboost_family_none | secops/detection-family | 66a8b919 |
| lightgbm_none_heldout_friday | secops/detection-heldout | a174088a |
| lightgbm_none_withport | secops/detection-ablations | d10efe46 |
| lightgbm_none_attempted_drop | secops/detection-ablations | 0cc2305b |

Registered models: `secops-detector` v1 (alias `champion`, run dcc36c43) and
`secops-family-classifier` v1 (alias `champion`, run 79e4b102). Completion check on 2026-10-03:
both reload through `registry.load_model` and reproduce their logged test PR-AUC (0.99993086) and
test macro-F1 (0.961262) to 1e-6, and the champion's artifacts resolve under
`$SECOPS_DATA_DIR/mlflow/artifacts/`.
