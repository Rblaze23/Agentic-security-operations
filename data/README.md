# Dataset: DistriNet-improved CIC-IDS-2017

Nothing under `data/` is committed except this file. The raw and processed data live under
`$SECOPS_DATA_DIR` (default `~/data/secops`), on the Linux filesystem.

## Three layers, kept distinct

| Layer | What it is | Who did it |
|---|---|---|
| Original CIC-IDS-2017 | 5 days of pcaps (3–7 July 2017), flows extracted with CICFlowMeter v3, labels from a time-and-IP schedule; ~2.83 M flows, 78 features | Canadian Institute for Cybersecurity, UNB (Sharafaldin, Lashkari, Ghorbani, ICISSP 2018) |
| DistriNet improved | Same pcaps re-extracted with a corrected CICFlowMeter fork, relabelled with published per-attack rules, "Attempted" sub-labels added; 2,099,976 flows, 82 features + 9 identifier/label columns | KU Leuven DistriNet (Engelen, Rimmer, Joosen, WTMC 2021; Liu, Engelen, Lynar, Essam, Joosen, IEEE CNS 2022) |
| Our preprocessing | Identifier removal, infinity handling, Attempted policy, exact-duplicate removal, family grouping, chronological splitting, Parquet build (`secops-data build`) | this repository; see `docs/dataset.md` |

## Source

- Download page: https://intrusion-detection.distrinet-research.be/CNS2022/Dataset_Download.html
- Archive: https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip
  (343,549,013 bytes, server timestamp 2023-04-27; SHA-256
  `97fdb91d339e2d8cf5627f981b831e5e7e400b981c58181c451a38fd03c48883`)
- Labelling code: https://github.com/GintsEngelen/CNS2022_Code
- Corrected flow extractor: https://github.com/GintsEngelen/CICFlowMeter
- Original dataset page and testbed description: https://www.unb.ca/cic/datasets/ids-2017.html

## Expected files (measured 2026-10-03)

| File | Bytes | Rows | SHA-256 |
|---|---|---|---|
| monday.csv | 207,875,155 | 371,624 | `51fe5dc962626efb4ae70dce0303072fb780da0932822b651202ee9c2fbc1aff` |
| tuesday.csv | 178,397,720 | 322,078 | `e2a0a5b631dfc6b455cc9f9a88b944110637d70a7f74171473925f76f38b6b0c` |
| wednesday.csv | 291,290,505 | 496,641 | `bf46c5f3c792e8817381f724511229569606918eaf07ac986d7a2592b6341bc2` |
| thursday.csv | 189,519,159 | 362,076 | `78a4d11eaf473d099e30e71ddb01e0f38218e844c0a9cdd36602145d674af482` |
| friday.csv | 285,188,226 | 547,557 | `ebd499e6f23bd59f9cb81bec28178491b02b925fa5640a24215c9437d79482d0` |

`secops-data download` refuses files whose size or hash differs from this table; the manifest
lives in `src/secops/data/manifest.py`.

## Terms of use

Neither the CIC page nor the DistriNet page publishes a formal licence text. Both ask for citation.
This project treats the data as **research-use only**: it is never redistributed, never committed
(the `.gitignore` excludes `data/raw`, `data/processed`, `*.csv` under those paths, `*.zip`,
`*.parquet`), and both papers are cited wherever results appear. The only derived data in git is a
sampled test fixture of under 1,000 rows (`tests/fixtures/mini_cicids/`), documented there.

```bibtex
@inproceedings{sharafaldin2018toward,
  title     = {Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization},
  author    = {Sharafaldin, Iman and Lashkari, Arash Habibi and Ghorbani, Ali A.},
  booktitle = {Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)},
  year      = {2018}
}
@inproceedings{liu2022error,
  title        = {Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018},
  author       = {Liu, Lisa and Engelen, Gints and Lynar, Timothy and Essam, Daryl and Joosen, Wouter},
  booktitle    = {2022 IEEE Conference on Communications and Network Security (CNS)},
  pages        = {254--262},
  year         = {2022},
  organization = {IEEE}
}
```

## Commands

```bash
export SECOPS_DATA_DIR=$HOME/data/secops   # or put it in .env
make data-download   # uv run secops-data download  (fetch, verify, extract, verify)
make data-build      # uv run secops-data build --attempted-policy relabel_benign (and drop)
```
