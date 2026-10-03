# Dataset

Every number in this document was measured by code in this repository on 2026-10-03 (`secops-data
build`, reports under `$SECOPS_DATA_DIR/reports/`). Nothing is copied from papers unless attributed.

## 1. Source and citations

The project uses **CIC-IDS-2017 in the corrected ("improved") version published by the DistriNet
group at KU Leuven**. Three layers are kept apart throughout the documentation because each one
changed the data:

| Layer | What it is | Who did it |
|---|---|---|
| Original CIC-IDS-2017 | Five days of packet captures (3–7 July 2017) on a documented testbed, flows extracted with CICFlowMeter v3, labels assigned from a time-and-IP schedule. About 2.83 M flows, 78 features. | Canadian Institute for Cybersecurity, University of New Brunswick. Sharafaldin, Lashkari, Ghorbani, *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*, ICISSP 2018. |
| DistriNet improved | The same captures re-extracted with a corrected CICFlowMeter fork and relabelled with published, per-attack rules; "Attempted" sub-labels added for attacker flows that carried no malicious payload. 2,099,976 flows, 82 features plus 9 identifier/label columns. | Engelen, Rimmer, Joosen, *Troubleshooting an Intrusion Detection Dataset: the CICFlowMeter and its flaw*, WTMC 2021; Liu, Engelen, Lynar, Essam, Joosen, *Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018*, IEEE CNS 2022. |
| This repository | Identifier removal, infinity handling, Attempted policy, exact-duplicate removal, family grouping, chronological splitting, Parquet build. | `src/secops/data/` |

Links: original page `https://www.unb.ca/cic/datasets/ids-2017.html`; improved data
`https://intrusion-detection.distrinet-research.be/CNS2022/`; labelling code
`https://github.com/GintsEngelen/CNS2022_Code`; corrected extractor
`https://github.com/GintsEngelen/CICFlowMeter`. BibTeX entries are in `data/README.md`.

Terms: neither page publishes a licence text; both require citation. The data is treated as
research-use only, is never redistributed and is never committed. The one derived artefact in git
is a 986-row sampled test fixture (`tests/fixtures/mini_cicids/`).

Why the corrected variant: the original extractor terminated TCP flows on the first FIN, ignored
RST, mis-counted PSH/URG flags and encoded absolute timestamps into the Active/Idle features (a
leakage channel). The fork fixes those; the relabelling fixes mis-timed attacks (SQL injection
timestamps were five minutes off, Hulk used `Connection: close`, an entire infiltration-phase port
scan was labelled benign). The rules are published, so the corrections are auditable.

## 2. Acquisition and verification

```bash
export SECOPS_DATA_DIR=$HOME/data/secops      # Linux filesystem, never under /mnt/d
uv run secops-data download                    # fetch zip, verify, extract, verify CSVs
uv run secops-data verify                      # re-check an existing copy
```

The archive is openly served (no form). `download` checks the zip SHA-256
(`97fdb91d339e2d8cf5627f981b831e5e7e400b981c58181c451a38fd03c48883`, 343,549,013 bytes), extracts the
five CSVs and checks each against the manifest in `src/secops/data/manifest.py`:

| File | Bytes | Rows | SHA-256 |
|---|---|---|---|
| monday.csv | 207,875,155 | 371,624 | `51fe5dc9…c9bc1aff` |
| tuesday.csv | 178,397,720 | 322,078 | `e2a0a5b6…38b6b0c` |
| wednesday.csv | 291,290,505 | 496,641 | `bf46c5f3…6341bc2` |
| thursday.csv | 189,519,159 | 362,076 | `78a4d11e…74af482` |
| friday.csv | 285,188,226 | 547,557 | `ebd499e6…9482d0` |

Full hashes are in `data/README.md`. A digest of the five hashes is logged as the MLflow tag
`manifest_sha` on every training run, together with the git commit, so any reported number can be
traced to exact bytes and code.

## 3. Structure

91 columns per row: `id` (row index), `Flow ID`, `Src IP`, `Src Port`, `Dst IP`, `Dst Port`,
`Protocol`, `Timestamp` (UTC with microseconds), 81 flow statistics from `Flow Duration` to
`Total TCP Flow Time`, `Label`, and `Attempted Category` (-1 = not attempted, 0–6 = reason code).

Timestamps are reliable and in UTC: the FTP brute force runs 12:19–13:20 UTC, which is 9:19–10:20
Atlantic Daylight Time, matching the published schedule. This is what makes a chronological split
possible; the original CSVs had ambiguous 12-hour timestamps.

**Model features (82):** `Protocol` plus the 81 statistics, including the fork's new
`Fwd RST Flags`, `Bwd RST Flags`, `ICMP Code`, `ICMP Type`, `Total TCP Flow Time`. All numeric.
Identifier columns are never features (section 5). `Dst Port` is excluded from the default feature
set and measured as an ablation.

### 3.1 Labels as delivered (before any processing)

| Label | Rows | "Attempted" rows (separate label in the file) |
|---|---|---|
| BENIGN | 1,582,566 | — |
| Portscan | 159,066 | 0 |
| DoS Hulk | 158,468 | 581 |
| DDoS | 95,144 | 0 |
| Infiltration - Portscan | 71,767 | 0 |
| DoS GoldenEye | 7,567 | 80 |
| FTP-Patator | 3,972 | 12 |
| DoS Slowloris | 3,859 | 1,847 |
| SSH-Patator | 2,961 | 27 |
| DoS Slowhttptest | 1,740 | 3,368 |
| Botnet | 736 | 4,067 |
| Web Attack - Brute Force | 73 | 1,292 |
| Infiltration | 36 | 45 |
| Web Attack - XSS | 18 | 655 |
| Web Attack - SQL Injection | 13 | 5 |
| Heartbleed | 11 | 0 |

Attempted rows total 11,979. Testbed facts from the CIC page: the external attacker is
205.174.165.73 (Kali), the internal attacker address seen in flows is 172.16.0.1, the main victim is
the Ubuntu web server 192.168.10.50, and 192.168.10.8 is the infiltrated host that performed the
internal port scan.

### 3.2 Targets

- **Binary** `is_attack`: 1 for every non-BENIGN label after the Attempted policy (section 4.3).
- **Family** (attack rows only), grouped so each class is large enough to measure:

| Family | Labels | Rows before dedup | Rows after dedup |
|---|---|---|---|
| `dos` | DoS Hulk, GoldenEye, Slowloris, Slowhttptest | 171,634 | 171,634 |
| `port_scan` | Portscan, Infiltration - Portscan | 230,833 | 7,810 |
| `ddos` | DDoS | 95,144 | 95,144 |
| `brute_force` | FTP-Patator, SSH-Patator | 6,933 | 6,933 |
| `botnet` | Botnet | 736 | 736 |
| `web_attack` | Web Brute Force, XSS, SQL Injection | 104 | 104 |
| `rare_exploit` | Heartbleed, Infiltration | 47 | 47 |

`rare_exploit` is excluded from family-classifier training and scoring (47 rows cannot be learned
or evaluated) but stays in the binary task as attack. `web_attack` is trained and its metrics are
flagged as low-support. `Infiltration - Portscan` maps to `port_scan` because the flows are an
internal host's port sweep; that the source is internal is context an enrichment tool should
surface, not something flow statistics encode. Original labels are preserved in `label_raw`.

## 4. Data-quality findings

### 4.1 Exact duplicates: 385,020 rows removed (18.3%)

Rows identical on all 82 features and the label are removed before splitting, keeping the first
occurrence. Measured per label (`relabel_benign` policy):

| Label | Duplicates removed | Rows before | Rows after | Share removed |
|---|---|---|---|---|
| BENIGN (incl. relabelled Attempted) | 161,997 | 1,594,545 | 1,432,548 | 10.2% |
| Portscan | 157,383 | 159,066 | 1,683 | 98.9% |
| Infiltration - Portscan | 65,640 | 71,767 | 6,127 | 91.5% |
| every other label | 0 | — | — | 0% |

Port-scan probes are identical SYN packets to different ports; once the destination port is removed
from the feature set there is nothing to distinguish them, so a scan of 65,000 ports becomes a
handful of unique feature vectors. Keeping the copies would inflate every port-scan metric (copies
would land on both sides of any split) and over-weight the family. The cost is that `port_scan`
drops to 7,810 training-usable rows and that models never see scan *volume*; a per-source burst
count is exactly the context the Phase 3 event store and tools provide instead.

The per-file count of duplicates (312,810) is lower than the global count because benign flows
repeat across days and because relabelling Attempted flows to BENIGN creates further identical
rows.

### 4.2 Infinity and missing values

10 cells contain `±Infinity` (`Flow Bytes/s` and `Flow Packets/s` on zero-duration flows: 6 in
monday.csv, 2 in wednesday.csv, 2 in thursday.csv). They become NaN; gradient-boosted trees handle
NaN natively and logistic regression receives median imputation fitted on the training split only.
No column has NaN in the delivered files, no column is constant across the five files, and no
feature column is non-numeric. (The original CSVs had NaNs and a duplicated `Fwd Header Length`
column; the fork removed both.)

### 4.3 "Attempted" flows: 11,979 rows

The dataset authors label attacker-generated flows that exhibit no malicious behaviour as
`<attack> - Attempted` with a category code: 0 no payload sent, 1 port/system closed, 2 attack
startup/teardown, 3 no malicious payload, 4 attack artefacts mixed with legitimate traffic, 5
attack implemented incorrectly, 6 target unresponsive. Their instruction, quoted from the download
page:

> Under no circumstance should the 'Attempted' flows be treated as a separate label for your
> Machine Learning model!

and their recommendation is to relabel them as benign. This repository implements two policies,
recorded on every MLflow run as the tag `attempted_policy`:

- `relabel_benign` (default, the authors' recommendation): the flow stays, labelled BENIGN.
- `drop` (ablation): the flow is removed before splitting.

Both keep `label_raw` and `Attempted Category`, so an event store can still show, for example, the
4,067 post-shutdown connection attempts from the botnet victims to the attacker's command server.

### 4.4 What the corrected labels changed

Compared with the original dataset: confirmed web attacks shrink from roughly 2,180 to 104 flows
(most of the original "web attack" rows were startup, teardown or GET requests without a
payload); Botnet has 736 live flows plus 4,067 attempted; Infiltration is 36 flows plus a
71,767-flow internal port scan that the original labelled benign; DoS Hulk in the original used
`Connection: close`, so the fork relabels it by packet signature rather than by time window alone.
Per-attack notes are on the DistriNet CIC-IDS-2017 page.

### 4.5 Active/Idle sanity

The original extractor encoded absolute timestamps into the Active/Idle features. In the improved
files the maximum Active/Idle value is 120 s (the flow timeout), verified in the split report.

## 5. Leakage risks and controls

| Risk | Channel | Control |
|---|---|---|
| Identifier leakage | `Src IP`, `Dst IP`, `Flow ID` identify attacker and victim machines | Never features; kept as metadata only. `Flow ID` is not even loaded. |
| Port leakage | Attack `Dst Port` is one of {21, 22, 80, 444, 8080} in this testbed | Excluded from the default feature set (`v1-noport`); measured as the `v1-withport` ablation and reported as a caveat. |
| Temporal leakage | A random split puts neighbouring flows of one burst on both sides | Chronological split inside each (day, label) group; a leak check in the build asserts no validation/test timestamp precedes a training timestamp within a group. |
| Duplicate leakage | Identical rows on both sides of a split | Exact-duplicate removal before splitting (section 4.1). |
| Preprocessing leakage | Imputer or scaler fitted on all data | Fitted inside the sklearn `Pipeline`, on the training split only. |
| Threshold leakage | Operating point tuned on the test split | Tuned on validation; test is scored once per configuration. |
| Label-encoded time | Active/Idle features encoding absolute time | Fixed by the fork; verified (section 4.5). |
| Row-order leakage | `id` correlates with time and day | Never a feature. |

Known optimism source, documented rather than hidden: the validation split is used both for early
stopping and for threshold selection. The test split is untouched by either.

## 6. Splits

Two assignments are written into the processed Parquet as columns, so one build serves every
experiment.

**Primary, `split_chrono` (chronological within group).** For each (day, label-after-policy)
group, rows are sorted by `Timestamp` then `id`; the first 70% go to train, the next 15% to
validation, the last 15% to test. Every class is present in every split, no future flow is in train,
and each attack's test rows are the *end* of that attack. Measured counts (`relabel_benign`):

| Family | Train | Validation | Test |
|---|---|---|---|
| benign | 1,002,782 | 214,882 | 214,884 |
| dos | 120,142 | 25,745 | 25,747 |
| ddos | 66,600 | 14,272 | 14,272 |
| port_scan | 5,466 | 1,171 | 1,173 |
| brute_force | 4,852 | 1,040 | 1,041 |
| botnet | 515 | 110 | 111 |
| web_attack | 72 | 16 | 16 |
| rare_exploit | 32 | 7 | 8 |
| **total** | **1,200,461** | **257,243** | **257,252** |

**Secondary, `split_heldout` (held-out Friday).** Train and validation come from Monday–Thursday
(chronological within group, 85/15); test is all of Friday. Botnet and DDoS are absent from
training; Friday's external port scan is a behaviour related to Thursday's internal one. Binary
task only; measures generalisation to attacks the model has never seen.

| Family | Train (Mon–Thu) | Validation (Mon–Thu) | Test (Friday) |
|---|---|---|---|
| benign | 1,006,043 | 177,538 | 248,967 |
| dos | 145,887 | 25,747 | — |
| brute_force | 5,892 | 1,041 | — |
| port_scan | 5,207 | 920 | 1,683 |
| web_attack | 88 | 16 | — |
| rare_exploit | 39 | 8 | — |
| ddos | — | — | 95,144 |
| botnet | — | — | 736 |
| **total** | **1,163,156** | **205,270** | **346,530** |

**Rejected: random stratified split.** Flows inside one attack burst are near-identical; a random
split places siblings on both sides and produces the 99.9% scores common in papers on this dataset.
It was never run here, so no inflated number exists to be quoted.

## 7. Build pipeline

```
raw CSV (per day) ──read_day──▶ typed frame (float32 features, UTC timestamps, 'day')
   │   header must equal the 91 expected columns exactly, otherwise SchemaError
   ▼
concat ──clean(policy)──▶ label_raw kept; label = base label or BENIGN (policy);
   │                       family, is_attack derived; ±inf → NaN; exact duplicates dropped
   ▼
assign_chrono_within_group ─▶ split_chrono ; assign_heldout_day ─▶ split_heldout
   │   check_no_time_leak on both (raises LeakError)
   ▼
processed/<policy>/flows.parquet  +  reports/<policy>/{cleaning_report,split_report}.json
```

Measured on the full data: 52 s wall-clock and 5.1 GB peak resident memory per policy on an
8-core WSL2 machine with 8.7 GB RAM. Output: 1,714,956 rows (`relabel_benign`) and 1,705,865 rows
(`drop`).

## 8. Limitations

- One testbed, one week, 2017 traffic; benign traffic is profile-generated. Scores are not
  transferable to another network without re-measurement.
- Residual label noise remains even after the corrections (the authors document cases they could
  not isolate, such as the infiltration payload download).
- Deduplication removes scan volume from the training data; a detector trained here recognises the
  *shape* of a probe, not the *count* of probes, which the event-store tools must supply.
- `web_attack` (104 rows) and `rare_exploit` (47 rows) are too small for reliable per-class
  numbers; they are reported with that caveat or excluded.
- Only the DistriNet variant is wired up. Running the official CIC CSVs through the same pipeline
  requires a second manifest and header map and is listed as future work.
