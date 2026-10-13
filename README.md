# Digital Government Initiatives and Public Service Delivery in India

QM640 Data Analytics Capstone · Walsh College

This repository contains the data, code and instructions for a state-level panel study of whether
the adoption of digital public services is associated with timelier, more inclusive service delivery
in India. The unit of analysis is the state/UT-year: 36 units × 2018–2025. The planned analytic
sample is at least 30 units × 8 years = 240 state-years.

| RQ | Question | Core method | Minimum N |
|---|---|---|---|
| RQ1 (core) | Is digital adoption associated with timely MGNREGS wage payment within states over time? | Two-way fixed-effects OLS, state-clustered SEs | 216 state-years |
| RQ2 (core) | Does that association differ by state literacy and income? | Moderated fixed-effects regression (2 interactions) | 213 state-years |
| RQ3 (secondary, gated) | Is online citizen sentiment about digital services associated with state digital adoption? | VADER sentiment, Spearman correlation, cluster bootstrap | 102 state-years; 505 labelled posts |
| RQ4 (core) | Is digital adoption associated with the inclusion of women and rural residents? | Fixed-effects regression, Holm-adjusted | 206 state-years |

`python -m src.models.power_analysis` reproduces every sample-size figure.

## Setup

```bash
git clone https://github.com/<your-username>/qm640-digital-governance.git
cd qm640-digital-governance
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # pinned versions; Python 3.10+
python run_all.py                  # download PhonePe data, load filled templates, rebuild panel
python run_all.py --offline        # same, reusing the existing PhonePe download
python run_all.py --offline --models   # also estimate RQ1-RQ4 -> reports/model_results.md
```

Each model runs only when its analysis sample reaches the minimum N. Until then, the report
says what is missing.

## Outputs

| File | Contents |
|---|---|
| `data/processed/state_year_panel.csv` | All sources merged, one row per state-year |
| `data/processed/analysis_panel.csv` | Panel plus per-capita measures, logs, indices, sentiment, sample flags |
| `data/processed/data_dictionary.csv` | Every column: description, unit, source, role, type, % non-missing (generated; fails if any column is undocumented) |
| `config/data_dictionary_posts.csv` | Post-level fields and sentiment labels (RQ3) |
| `reports/missing_data_report.md` | Coverage by variable and year, and value-status counts |
| `reports/validation/<source>.txt` | Problems found in each filled template |
| `reports/model_results.md` | RQ1–RQ4 estimates and rolling-origin validation |
| `reports/sentiment_validation.txt` | Classifier accuracy, macro-F1 and kappa |
| `notebooks/01_eda_phonepe.ipynb` | Exploratory analysis of the PhonePe data |

## Data collection status

| Source | Variables | Collection | Status |
|---|---|---|---|
| PhonePe Pulse (GitHub) | `upi_txn` (+ P2P / retail / utility), `upi_users`, `upi_merchants` | Automated (`src/collect/phonepe_pulse.py`) | **Done**: 288/288 state-years |
| National Commission on Population projections | `population` | Template | To do (first) |
| MGNREGA MIS | `nrega_timely_pct` (primary outcome), `nrega_women_pd_pct` | Template / data.gov.in API | To do |
| Annavitran (PDS) | `pds_epos_auth_pct` | Template | To do |
| e-Taal | `etaal_txn` | Template | To do |
| DBT Bharat | `dbt_amount_cr` | Template / data.gov.in API | To do |
| TRAI | `internet_per100` | Template + `config/trai_service_areas.csv` | To do |
| RBI Handbook of Statistics on Indian States | `literacy_pct` (Census 2011), `nsdp_pc_real` | Template (Excel tables) | To do |
| PMJDY | `pmjdy_women_pct`, `pmjdy_rural_pct` | Template | To do |
| NeSDA | `nesda_score` | Template (2019, 2021, 2023 reports) | Optional (descriptive) |
| Reddit Data API | post text → sentiment | `src/collect/social_posts.py`, then `src/features/sentiment.py` | Gated: pilot pull by week 3 |

`config/manual_sources.csv` gives the exact report, unit and plausible range for every variable.

### Filling a template

1. `python run_all.py` writes `templates/<source>.csv`: one row for each state × year that source covers.
2. Copy it to `data/raw/<source>/<source>.csv` and enter values from the official report.
   - Record FY 2018-19 as year 2018.
   - Save the report (PDF, XLSX or screenshot) in the same folder, and write its name in `source_document`.
3. Use these codes for anything that is not a number:

   | Code | Meaning | Treatment |
   |---|---|---|
   | *(blank)* | Not collected yet | Missing |
   | `NR` | Source does not publish the value for that year | Missing; listwise deletion; multiple imputation as a sensitivity check if the variable is >10% missing |
   | `NS` | Scheme/indicator does not operate in that unit (e.g. no rural MGNREGS) | Structural: flagged and excluded, never imputed |

4. Run `python run_all.py --offline` and read `reports/validation/<source>.txt`. The pipeline handles these cases:
   - **Unknown state names** are dropped and listed.
   - **Out-of-range values** are set to missing and listed.
   - **Pre-2020 rows for Dadra & Nagar Haveli and for Daman & Diu** are combined (summed for counts, averaged for rates).

### RQ3 sentiment workflow (only if the pilot pull clears the gate)

```bash
export REDDIT_CLIENT_ID=... REDDIT_CLIENT_SECRET=... REDDIT_USER_AGENT="qm640 research"
pip install -r requirements-optional.txt
python -m src.collect.social_posts reddit
python -m src.features.sentiment score      # VADER, English/romanised posts 2023-2025
python -m src.features.sentiment sample     # 505 posts to hand-label (60 double-coded)
python -m src.features.sentiment evaluate   # accept if macro-F1 >= .70 and kappa >= .60
python -m src.features.sentiment aggregate  # state-year net sentiment (>= 30 posts)
python -m src.features.sentiment purge      # delete raw text at project end
```

Ethics: no usernames are collected. Raw text is git-ignored and deleted at the end of the project.
Only post IDs and state-year aggregates are shared, under the Reddit Data API terms for
non-commercial research.

## Layout

```
config/          states.csv, manual_sources.csv, trai_service_areas.csv
templates/       blank entry sheets (regenerated each run)
data/raw/        source files and evidence; PhonePe clone and social posts are git-ignored
data/interim/    one cleaned file per source + _status_<source>.csv (value status)
data/processed/  state_year_panel.csv, analysis_panel.csv, data_dictionary.csv
src/collect/     phonepe_pulse.py, manual_sources.py, datagov_api.py, social_posts.py
src/clean/       build_panel.py
src/features/    build_features.py, sentiment.py
src/models/      power_analysis.py, rq_models.py
notebooks/       01_eda_phonepe.ipynb
reports/         missing-data report, validation logs, model results, figures
run_all.py       full pipeline
```

## Versioning and reproducibility

- **Python environment:** versions are pinned in `requirements.txt`. Optional extras are in `requirements-optional.txt`.
- **PhonePe data:** the commit used is recorded in `data/raw/phonepe_pulse_COMMIT.txt`, currently `257bf38e30` from 24 Sep 2026.
- **Milestones:** tag the repository at each one, for example `git tag v0.1-synopsis` and `v0.2-data-complete`.
- **Randomness:** random steps (labelling sample, bootstrap) use seed 640.
