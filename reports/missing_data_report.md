# Missing-data report

Panel: 36 states/UTs x 8 years = 288 rows. Values are % of the 36 units with data in that year.

Variables at 0% have not been collected yet; see README > Data collection status.

| Variable | Source | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | Overall |
|---|---|---|---|---|---|---|---|---|---|---|
| upi_txn_p2p | phonepe | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100.0 |
| upi_txn_retail | phonepe | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100.0 |
| upi_txn_utility | phonepe | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100.0 |
| upi_txn | phonepe | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100.0 |
| upi_users | phonepe | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100.0 |
| upi_merchants | phonepe | 92 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 99.0 |
| population | population | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100.0 |
| etaal_txn | etaal | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| dbt_amount_cr | dbt | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| nrega_timely_pct | nrega | 83 | 83 | 83 | 83 | 83 | 83 | 83 | 83 | 83.3 |
| nrega_women_pd_pct | nrega | 0 | 0 | 0 | 83 | 83 | 83 | 83 | 83 | 52.1 |
| nrega_persondays_lakh | nrega | 0 | 0 | 0 | 86 | 86 | 86 | 83 | 83 | 53.1 |
| nrega_efms_pct | nrega | 0 | 0 | 0 | 83 | 83 | 83 | 83 | 83 | 52.1 |
| nrega_timely_dash_pct | nrega | 0 | 0 | 0 | 83 | 83 | 83 | 83 | 83 | 52.1 |
| pds_epos_auth_pct | annavitran | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| nesda_score | nesda | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| internet_per100 | trai | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100.0 |
| literacy_pct | rbi | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100.0 |
| nsdp_pc_real | rbi | 92 | 92 | 92 | 92 | 92 | 69 | 0 | 0 | 66.0 |
| pmjdy_women_pct | pmjdy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| pmjdy_rural_pct | pmjdy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |

## Value status in filled templates

ok = numeric value; not_reported (NR) = not published for that year; not_applicable (NS) = scheme absent, structurally excluded; blank = not collected yet.

| Variable | not_applicable | not_reported | ok |
|---|---|---|---|
| internet_per100 | 0 | 0 | 288 |
| literacy_pct | 0 | 0 | 288 |
| nrega_efms_pct | 23 | 115 | 150 |
| nrega_persondays_lakh | 23 | 112 | 153 |
| nrega_timely_dash_pct | 23 | 115 | 150 |
| nrega_timely_pct | 24 | 24 | 240 |
| nrega_women_pd_pct | 23 | 115 | 150 |
| nsdp_pc_real | 0 | 98 | 190 |
| population | 0 | 0 | 288 |

## Boundary changes handled by flags

- `flag_jk_incl_ladakh`: J&K values before 2019 include Ladakh.
- `flag_ladakh_not_separate`: Ladakh had no separate figures before 2019 in most sources.
- `flag_dh_premerger`: Dadra & Nagar Haveli and Daman & Diu combined before the 2020 merger.

PhonePe state-years without a full year of data: 0
