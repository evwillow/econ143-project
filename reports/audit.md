# M0 — Survivorship & Bad-Bar Audit

_Generated 2026-05-03T01:10:24+00:00_

- Daily bars source: `C:\Users\evanm\Documents\courses\Econ143\ec143-project\data\raw\stocks\daily`
- Window: 2010-2025

## 1. Survivorship Check

5 of 5 known-delisted probe tickers present (100.0000%). Decision gate: **proceed**.

| Ticker | Present | Bar count | First bar | Last bar |
|---|---|---|---|---|
| LEH | yes | 1,265 | 2003-09-10 | 2008-09-17 |
| BBBY | yes | 5,009 | 2003-09-10 | 2026-02-19 |
| WCG | yes | 3,863 | 2004-07-01 | 2020-01-23 |
| SIVB | yes | 4,752 | 2003-09-10 | 2023-03-09 |
| FRC | yes | 4,225 | 2003-09-10 | 2023-04-28 |

Decision gate (per spec): >=50% present -> proceed with caveat; <20% present -> pivot to CRSP.

## 2. Bad-Bar Check

Total bars in 2010-2025: **35,343,096**. Hard-bad (a+b+c+d): 1,218,314 (3.4471%).

| Category | Count | Fraction |
|---|---:|---:|
| (a) OHLC inconsistent | 0 | 0.0000% |
| (b) Null OHLCV | 0 | 0.0000% |
| (c) Non-positive price | 0 | 0.0000% |
| (d) Stale feed (O==H==L==C, V>0) | 1,218,314 | 3.4471% |
| (e) Suspicious move >50% (informational) | 38,438 | 0.1088% |

_Note: splits.parquet not present; suspicious moves not cross-referenced._

Per-year hard-bad fraction (flag threshold: >0.1000%):

| Year | Bars | Hard-bad | Fraction | Flagged |
|---:|---:|---:|---:|---|
| 2010 | 1,876,075 | 48,869 | 2.6049% | YES |
| 2011 | 1,879,131 | 47,994 | 2.5541% | YES |
| 2012 | 1,849,358 | 56,856 | 3.0744% | YES |
| 2013 | 1,874,439 | 47,188 | 2.5174% | YES |
| 2014 | 1,938,492 | 43,543 | 2.2462% | YES |
| 2015 | 1,979,385 | 50,055 | 2.5288% | YES |
| 2016 | 1,984,727 | 54,508 | 2.7464% | YES |
| 2017 | 2,003,087 | 59,762 | 2.9835% | YES |
| 2018 | 2,048,018 | 65,871 | 3.2163% | YES |
| 2019 | 2,134,905 | 70,719 | 3.3125% | YES |
| 2020 | 2,217,421 | 59,528 | 2.6846% | YES |
| 2021 | 2,620,217 | 86,785 | 3.3121% | YES |
| 2022 | 2,795,047 | 159,684 | 5.7131% | YES |
| 2023 | 2,663,345 | 138,601 | 5.2040% | YES |
| 2024 | 2,665,129 | 116,233 | 4.3613% | YES |
| 2025 | 2,814,320 | 112,118 | 3.9838% | YES |

**Flagged years:** 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025

## 3. Coverage Check

Distinct trading dates per calendar year (flag threshold: < 200, expected ~252):

| Year | Trading dates | Flagged |
|---:|---:|---|
| 2010 | 252 |  |
| 2011 | 252 |  |
| 2012 | 250 |  |
| 2013 | 252 |  |
| 2014 | 252 |  |
| 2015 | 252 |  |
| 2016 | 252 |  |
| 2017 | 251 |  |
| 2018 | 251 |  |
| 2019 | 252 |  |
| 2020 | 253 |  |
| 2021 | 252 |  |
| 2022 | 251 |  |
| 2023 | 250 |  |
| 2024 | 252 |  |
| 2025 | 250 |  |

No years flagged.
