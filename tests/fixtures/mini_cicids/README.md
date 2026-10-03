# mini_cicids test fixture

A deterministic sample of the DistriNet-improved CIC-IDS-2017 CSVs used by the unit and
integration tests: per day, up to 60 BENIGN rows and up to 30 rows of every other label, rows
kept in their original order with the original 91 columns. Under 1,000 rows in total.

The sample is derived from research-use data; it is here only so that the tests exercise the real
column layout, timestamps and label vocabulary. Cite the dataset papers if you reuse it (see
`data/README.md` for the BibTeX entries: Sharafaldin et al., ICISSP 2018; Liu et al., IEEE CNS 2022).

Regenerate with:

```bash
SECOPS_DATA_DIR=$HOME/data/secops uv run python tests/fixtures/make_fixture.py
```
