# M1 — Universe & Breakout Detection Validation

- Feature table: 38,810,782 rows (30,514 tickers)
- Tickers split into >1 segment by the 60d recycling gap: **9,660** (31.66%)
- Strict total setups: **10,772**
- Loose total setups:  **32,659**

## Setups by year x variant

| Year | strict | loose |
|---:|---:|---:|
| 2010 | 312 | 954 |
| 2011 | 816 | 2,105 |
| 2012 | 257 | 679 |
| 2013 | 434 | 894 |
| 2014 | 545 | 1,663 |
| 2015 | 561 | 1,424 |
| 2016 | 316 | 849 |
| 2017 | 603 | 1,425 |
| 2018 | 884 | 2,226 |
| 2019 | 522 | 1,525 |
| 2020 | 895 | 2,925 |
| 2021 | 951 | 3,969 |
| 2022 | 965 | 3,683 |
| 2023 | 745 | 2,106 |
| 2024 | 1,094 | 2,837 |
| 2025 | 872 | 3,395 |

## Top 10 most-frequent setup tickers per variant

### strict

| Ticker | Setups |
|---|---:|
| BMA | 42 |
| CVNA | 34 |
| TDS | 30 |
| SHOP | 30 |
| RGEN | 29 |
| LGND | 29 |
| PRTA | 28 |
| TGTX | 28 |
| IOT | 27 |
| FOLD | 26 |

### loose

| Ticker | Setups |
|---|---:|
| STMP | 74 |
| TGTX | 66 |
| SOXL | 65 |
| MESO | 63 |
| CVNA | 63 |
| HIMS | 62 |
| VNET | 62 |
| PACB | 60 |
| NVAX | 59 |
| CENX | 58 |

## Regime sanity check (strict)

- 2018: 884
- 2020: 895
- 2021: 951
- 2022: 965

- WARN: strict 2022 (965) > strict 2018 (884).

## Caveats

- **`is_common_stock` filter not applied**: the daily-bars schema (ticker, date, OHLCV, transactions) has no security-type field. The universe therefore includes ETFs, ADRs, units, warrants, and other non-common-stock issuers. M2/M3 should either join Polygon ticker metadata or accept the contamination as a documented limitation.
- Pivot rule uses close-based extrema (no intraday H/L) for unambiguous argmax/argmin. Will diverge from chart-based pivot detection that uses high/low wicks.

## 10 random sample setups (in `m1_sample_setups.csv`)

| Ticker | Date | Variant | mom_12_1 | mom_pct | base_dur | pullback_pct | dist_52w_high_pct | close |
|---|---|---|---:|---:|---:|---:|---:|---:|
| JKS | 2011-05-23 | loose | 1.0001 | 0.976 | 38 | 0.1700 | 0.3764 | 24.27 |
| NKTR | 2012-07-26 | loose | 0.2043 | 0.872 | 40 | 0.0905 | 0.0905 | 132.60 |
| URI | 2012-08-29 | loose | 0.5339 | 0.962 | 11 | 0.1476 | 0.3276 | 31.48 |
| TA | 2013-05-10 | strict | 0.7859 | 0.981 | 40 | 0.1364 | 0.1364 | 10.57 |
| REGN | 2013-10-30 | loose | 0.7521 | 0.939 | 30 | 0.0804 | 0.0804 | 291.17 |
| INO | 2014-07-15 | loose | 0.5235 | 0.934 | 25 | 0.1027 | 0.3652 | 116.40 |
| XME | 2022-08-29 | loose | 0.0735 | 0.808 | 17 | 0.0935 | 0.2141 | 51.75 |
| RYAN | 2023-01-06 | loose | 0.0708 | 0.809 | 31 | 0.0971 | 0.1135 | 41.00 |
| AUPH | 2023-12-04 | loose | 0.4410 | 0.949 | 10 | 0.1027 | 0.2812 | 8.82 |
| LGND | 2024-11-15 | strict | 0.6285 | 0.921 | 29 | 0.1400 | 0.1400 | 111.71 |
