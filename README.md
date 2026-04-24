# AlphaForge

[![CI](https://github.com/PeNgzzzzz/AlphaForge/actions/workflows/ci.yml/badge.svg)](https://github.com/PeNgzzzzz/AlphaForge/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/PeNgzzzzz/AlphaForge)](LICENSE)

AlphaForge is a modular quant research workbench for daily cross-sectional equity research.

The project is built to be technically conservative, reproducible, and easy to explain. The emphasis is on timing correctness, transparent assumptions, inspectable backtests, and lightweight research tooling rather than exaggerated institutional realism.

- Representative outputs and charts: [RESULTS.md](RESULTS.md)

## What AlphaForge Covers

- Daily OHLCV, benchmark return-series, symbol-metadata, corporate-actions, fundamentals, classifications, memberships, borrow-availability, and trading-calendar validation with explicit schema, duplicate checks, and conservative integrity rules.
- Research dataset construction with close-anchored features, forward-return labels, optional average true range, optional Garman-Klass volatility, optional Parkinson volatility, optional Rogers-Satchell volatility, optional Yang-Zhang volatility, optional realized-volatility family features, optional trailing rolling skew/kurtosis features, and optional benchmark-aware rolling beta/correlation features.
- Optional lagged universe filters for price, rolling volume, rolling dollar volume, and listing history.
- Reusable price signals: momentum, mean reversion, and trend, with optional within-date winsorization and z-score/rank normalization.
- Long-only and long-short portfolio construction with equal-weight or score-weight normalization.
- Conservative daily close-to-close backtesting with explicit signal delay, rebalance frequency, transaction costs, turnover limits, and max-position caps.
- Performance, risk, and factor diagnostics, including benchmark-relative metrics, IC, quantile analysis, and coverage diagnostics.
- Config-driven CLI workflows for validation, dataset building, backtesting, reporting, parameter sweeps, walk-forward evaluation, and experiment comparison.
- Static report visualization, HTML report packaging, and lightweight artifact bundles.

## Key Design Choices

- Daily data only: the current engine is intentionally close-to-close and date-based.
- Explicit timing: feature timing, signal timing, and execution timing are separated.
- Conservative defaults: benchmark alignment is exact, malformed data fail loudly, and artifact outputs stay simple files.
- Lightweight tooling: CSV, TOML, PNG, HTML, and `runs.csv` are preferred over heavier tracking infrastructure.

## Current Capabilities

### Data and Research

- CSV and Parquet OHLCV loading
- Canonical schema validation
- CSV and Parquet trading calendar loading with canonical date-only session normalization
- CSV and Parquet benchmark return-series loading with canonical `date` / `benchmark_return` normalization
- CSV and Parquet corporate-actions loading with canonical `symbol` / `ex_date` / `action_type` / `split_ratio` / `cash_amount` normalization
- CSV and Parquet long-form fundamentals loading with canonical `symbol` / `period_end_date` / `release_date` / `metric_name` / `metric_value` normalization
- CSV and Parquet sector/industry classifications loading with canonical `symbol` / `effective_date` / `sector` / `industry` normalization
- CSV and Parquet index membership loading with canonical `symbol` / `effective_date` / `index_name` / `is_member` normalization
- CSV and Parquet borrow availability loading with canonical `symbol` / `effective_date` / `is_borrowable` / `borrow_fee_bps` normalization
- Optional config-driven split-adjusted OHLCV loading with explicit backward price/volume adjustment factors
- CSV and Parquet symbol metadata loading with canonical `symbol` / `listing_date` / `delisting_date` normalization
- Deterministic sorting by `symbol` and `date`
- Forward returns, rolling volatility, and rolling average volume
- Optional next-session-safe fundamentals joins into the research dataset with explicit metric selection
- Optional effective-date-safe sector/industry classification joins into the research dataset with explicit field selection
- Optional effective-date-safe index membership joins into the research dataset with explicit index selection
- Optional effective-date-safe borrow availability joins into the research dataset with explicit field selection
- Optional trading calendar joins with fail-fast off-calendar date validation
- Optional trading-calendar validation for corporate-action `ex_date` values under `validate-data`
- Optional symbol metadata joins with fail-fast listing/delisting window validation
- Calendar-aware and metadata-aware listing-history counts for universe eligibility when those inputs are provided
- Lagged tradability-aware universe filtering with explicit eligibility diagnostics
- Optional within-date signal transforms with explicit `winsorize_quantile` and `cross_sectional_normalization` settings
- Optional trailing Garman-Klass volatility in the research dataset with an explicit window
- Optional trailing Parkinson volatility in the research dataset with an explicit window
- Optional trailing average true range in the research dataset with an explicit window
- Optional trailing Rogers-Satchell volatility in the research dataset with an explicit window
- Optional trailing Yang-Zhang volatility in the research dataset with an explicit window
- Optional trailing realized-volatility family features in the research dataset with explicit windows
- Optional trailing rolling skew/kurtosis features in the research dataset with explicit windows
- Optional exact-date benchmark rolling beta/correlation features in the research dataset with explicit trailing windows

### Portfolio and Backtest

- Long-only and long-short construction
- Equal-weight and score-weight allocation
- Optional `max_position_weight`
- Daily, weekly, and monthly rebalancing
- Split commission/slippage costs with legacy `transaction_cost_bps` compatibility
- Optional turnover caps with target vs realized execution diagnostics

### Analytics and Visualization

- Cumulative return, annualized return, volatility, Sharpe, drawdown, hit rate
- Benchmark-relative return, tracking error, and information ratio
- Rolling beta and rolling correlation versus a benchmark
- IC / Rank IC summaries
- Quantile bucket returns and top-bottom quantile spread diagnostics
- Signal coverage summary and coverage-through-time diagnostics
- Static PNG charts for NAV, drawdown, exposure/turnover, IC, cumulative IC, coverage, quantile diagnostics, and benchmark risk
- HTML report packaging that combines charts, headline summaries, and the plain-text report

### Tooling

- `validate-data`
- `build-dataset`
- `run-backtest`
- `report`
- `plot-report`
- `sweep-signal`
- `walk-forward-signal`
- `list-runs`
- `compare-runs`

## Repository Layout

```text
.
├── README.md
├── RESULTS.md
├── configs/
├── data/
├── artifacts/
├── src/
│   └── alphaforge/
│       ├── analytics/
│       ├── backtest/
│       ├── cli/
│       ├── common/
│       ├── data/
│       ├── features/
│       ├── portfolio/
│       ├── risk/
│       └── signals/
├── tests/
├── pyproject.toml
└── setup.py
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Quick Start

Validate data and build a research report:

```bash
./.venv/bin/alphaforge validate-data --config configs/stage4_flagship_example.toml
./.venv/bin/alphaforge report --config configs/stage4_flagship_example.toml
```

Write a full report artifact bundle:

```bash
./.venv/bin/alphaforge report --config configs/stage4_flagship_example.toml --artifact-dir artifacts/stage4_report
```

Write charts only:

```bash
./.venv/bin/alphaforge plot-report --config configs/stage4_flagship_example.toml --output-dir artifacts/stage4_charts
```

Run a parameter sweep and a walk-forward evaluation:

```bash
./.venv/bin/alphaforge sweep-signal --config configs/momentum_example.toml --parameter lookback --values 1 2 3
./.venv/bin/alphaforge walk-forward-signal --config configs/momentum_example.toml --parameter lookback --values 1 2 3 --train-periods 4 --test-periods 2
```

Compare indexed experiment runs:

```bash
./.venv/bin/alphaforge compare-runs --experiment-root artifacts/showcase_runs --limit 2 --artifact-dir artifacts/showcase_compare_report
```

## Bundled Example Configs

- `configs/momentum_example.toml`
- `configs/mean_reversion_example.toml`
- `configs/trend_example.toml`
- `configs/stage1_universe_example.toml`
- `configs/stage2_execution_example.toml`
- `configs/stage3_benchmark_example.toml`
- `configs/stage4_flagship_example.toml`

The example datasets in `data/raw/` are deterministic and synthetic. They are intended for reproducibility and pipeline inspection, not for claims about tradable alpha.

Example signal transform settings:

```toml
[signal]
name = "momentum"
lookback = 20
winsorize_quantile = 0.05
cross_sectional_normalization = "zscore"
```

These transforms are applied within each date only, after any lagged universe eligibility mask has already removed ineligible rows.

Example Garman-Klass-volatility settings:

```toml
[dataset]
garman_klass_volatility_window = 20
```

This dataset feature uses trailing `open` / `high` / `low` / `close` observations through the current close and applies the daily Garman-Klass variance proxy before taking the rolling square root of the positive window mean.

Example Parkinson-volatility settings:

```toml
[dataset]
parkinson_volatility_window = 20
```

This dataset feature uses only trailing `high` / `low` observations through the current close and applies the daily Parkinson variance proxy before taking the rolling square root of the window mean.

Example average-true-range settings:

```toml
[dataset]
average_true_range_window = 20
```

This dataset feature uses trailing daily true range, defined as `max(high - low, abs(high - close_{t-1}), abs(low - close_{t-1}))`, and writes the trailing window mean in price units.

Example normalized-average-true-range settings:

```toml
[dataset]
normalized_average_true_range_window = 20
```

This dataset feature uses the same trailing daily true-range definition as ATR, then divides the trailing ATR level by the same-day `close` so the output is a dimensionless range proxy rather than a price-unit series.

Example Amihud-illiquidity settings:

```toml
[dataset]
amihud_illiquidity_window = 20
```

This dataset feature uses trailing `abs(daily_return) / (close * volume)` observations through the current close and writes the trailing mean. Days with zero dollar volume are treated conservatively as unavailable rather than forced into finite illiquidity values.

Example dollar-volume-z-score settings:

```toml
[dataset]
dollar_volume_zscore_window = 20
```

This dataset feature uses same-day `log(close * volume)` compared with the trailing mean and sample standard deviation of the prior `window` log dollar-volume observations. The window must be at least 2, and the rolling baseline excludes the current day.

Example volume-shock settings:

```toml
[dataset]
volume_shock_window = 20
```

This dataset feature uses same-day `log(volume)` minus the trailing mean of the prior `window` log-volume observations. The rolling baseline excludes the current day, and zero-volume observations are treated as unavailable.

Example relative-volume settings:

```toml
[dataset]
relative_volume_window = 20
```

This dataset feature uses same-day `volume` divided by the trailing mean of the prior `window` daily volume observations. The denominator excludes the current day on purpose, so the baseline stays explicitly historical.

Example relative-dollar-volume settings:

```toml
[dataset]
relative_dollar_volume_window = 20
```

This dataset feature uses same-day `close * volume` divided by the trailing mean of the prior `window` daily dollar-volume observations. The denominator excludes the current day on purpose, so the baseline stays explicitly historical.

Example Rogers-Satchell-volatility settings:

```toml
[dataset]
rogers_satchell_volatility_window = 20
```

This dataset feature uses trailing `open` / `high` / `low` / `close` observations through the current close and applies the daily Rogers-Satchell variance proxy before taking the rolling square root of the window mean.

Example Yang-Zhang-volatility settings:

```toml
[dataset]
yang_zhang_volatility_window = 20
```

This dataset feature uses trailing overnight, open-to-close, and Rogers-Satchell OHLC components through the current close. It currently requires a window of at least 2 because the overnight and open-to-close variance terms use sample variance.

Example realized-volatility family settings:

```toml
[dataset]
realized_volatility_window = 20
```

This dataset feature uses trailing strategy `daily_return` observations through the current close and writes root-mean-square realized volatility plus downside/upside variants over the same window.

Example rolling higher-moments settings:

```toml
[dataset]
higher_moments_window = 20
```

This dataset feature uses only trailing strategy `daily_return` observations through the current close and currently rejects windows smaller than 4.

Example benchmark-aware rolling statistics settings:

```toml
[dataset]
benchmark_rolling_window = 20

[benchmark]
path = "benchmark.csv"
name = "S&P 500"
return_column = "benchmark_return"
rolling_window = 20
```

This dataset feature requires exact benchmark/date alignment and uses only trailing strategy `daily_return` plus same-day benchmark returns observable through the current close.

## Results and Artifacts

Representative report outputs are documented in [RESULTS.md](RESULTS.md).

The current artifact design is intentionally lightweight:

- `results.csv` stores tabular outputs
- `report.txt` stores the plain-text report
- `metadata.json` stores a workflow snapshot and compact summaries
- `index.html` is written for report artifacts
- `charts/manifest.json` plus `charts/*.png` store static visual outputs
- `runs.csv` provides a minimal experiment index for repeated sweeps and walk-forward runs

## Testing

Latest local validation for the current repository state:

```bash
./.venv/bin/python -m pytest -q
```

Result:

```text
297 passed
```

## Limitations

- Daily data only; no intraday timestamps or intraday execution modeling
- No market impact, borrow cost, queue position, or order book simulation
- No optimizer-based portfolio construction or richer exposure constraints
- Benchmark analysis is based on date-only simple return series, not constituent-level attribution
- Trading calendar support currently uses explicit date-only session lists, not multi-exchange or intraday session engines
- Corporate actions currently support split-adjusted OHLCV plus split/cash-dividend event contracts; cash dividends are still not applied to total-return or dividend-adjusted price series
- Fundamentals currently support a long-form release-date-aware contract plus next-session-safe dataset joins for explicitly selected metrics, but still do not model release-time-of-day, restatement lineage, or broader point-in-time reference joins
- Classifications currently support only effective-date-safe sector/industry histories; they do not yet cover more complex classification lineage
- Borrow availability currently supports only effective-date-safe borrowable/fee histories; it does not yet drive short-sale constraints, borrow costs, or richer securities-financing workflows
- Memberships currently support only effective-date-safe index membership histories; they do not yet model constituent weights, intraday membership timing, or broader reference-data lineage
- Cross-sectional signal transforms currently cover within-date winsorization plus z-score/rank normalization only; they do not yet cover sector-relative normalization, neutralization, or robust scaling stacks
- Dataset-level rolling statistics currently cover average true range, normalized average true range, Amihud illiquidity, relative dollar volume, Garman-Klass volatility, Parkinson volatility, Rogers-Satchell volatility, Yang-Zhang volatility, daily-return-based realized volatility families, trailing skew/kurtosis, and exact-date-aligned trailing beta/correlation versus a single benchmark; they do not yet cover richer range-based estimators, intraday volatility estimators, multi-benchmark features, or residualization pipelines
- Symbol metadata currently covers symbol-level listing/delisting dates only, not identifier-history workflows
- Visual outputs are static PNG/HTML artifacts, not interactive dashboards
- Artifact tracking remains intentionally file-based rather than database-backed
