# AlphaForge

[![CI](https://github.com/PeNgzzzzz/AlphaForge/actions/workflows/ci.yml/badge.svg)](https://github.com/PeNgzzzzz/AlphaForge/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/PeNgzzzzz/AlphaForge)](LICENSE)

AlphaForge is a modular quant research workbench for daily cross-sectional equity research.

The project is built to be technically conservative, reproducible, and easy to explain. The emphasis is on timing correctness, transparent assumptions, inspectable backtests, and lightweight research tooling rather than exaggerated institutional realism.

- Representative outputs and charts: [RESULTS.md](RESULTS.md)

## What AlphaForge Covers

- Daily OHLCV, benchmark return-series, symbol-metadata, corporate-actions, fundamentals, shares-outstanding, classifications, memberships, borrow-availability, trading-status, and trading-calendar validation with explicit schema, duplicate checks, and conservative integrity rules.
- Research dataset construction with close-anchored features, forward-return labels, optional fundamental valuation/quality/growth/stability features, optional effective-date market-cap features, optional average true range, optional Garman-Klass volatility, optional Parkinson volatility, optional Rogers-Satchell volatility, optional Yang-Zhang volatility, optional realized-volatility family features, optional trailing rolling skew/kurtosis features, and optional benchmark-aware rolling beta/correlation plus residual-return features.
- Optional lagged universe filters for price, rolling volume, rolling dollar volume, listing history, required index membership, and explicit trading status.
- Reusable price signals backed by inspectable factor definitions: momentum, mean reversion, and trend, with optional within-date transform definitions for winsorization, clipping, numeric exposure residualization, z-score, robust z-score, and rank normalization.
- Long-only and long-short portfolio construction with equal-weight or score-weight normalization.
- Conservative daily close-to-close backtesting with explicit signal delay, rebalance frequency, transaction costs, turnover limits, and max-position caps.
- Performance, risk, and factor diagnostics, including benchmark-relative metrics, IC, rolling IC, quantile analysis, and coverage diagnostics.
- Config-driven CLI workflows for validation, dataset building, backtesting, reporting, parameter sweeps, walk-forward evaluation, and experiment comparison.
- Static report visualization, HTML report packaging, and lightweight artifact bundles.

## Key Design Choices

- Daily data only: the current engine is intentionally close-to-close and date-based.
- Explicit timing: feature timing, signal timing, and execution timing are separated.
- Conservative defaults: benchmark alignment is exact, malformed data fail loudly, and artifact outputs stay simple files.
- Lightweight tooling: CSV, TOML, PNG, HTML, and `runs.csv` are preferred over heavier tracking infrastructure.
- Scoped workflow internals: config-driven data loading helpers, pipeline assembly helpers, validation report helpers, parameter-sweep workflow helpers, walk-forward workflow helpers, report context helpers, report package orchestration helpers, research metadata helpers, artifact writers, chart writers, report text/metadata/HTML helpers, and run-comparison helpers live outside the main workflow orchestration module. The CLI entrypoint imports those focused modules directly, while `alphaforge/cli/workflows.py` remains an explicit legacy compatibility surface instead of absorbing every reference-data, research-pipeline, validate-data reporting, sweep-signal, walk-forward-signal, report-context, report-packaging, research-metadata, file-output, chart-output, or compare-runs concern.

## Current Capabilities

### Data and Research

- CSV and Parquet OHLCV loading
- Canonical schema validation
- CSV and Parquet trading calendar loading with canonical date-only session normalization
- CSV and Parquet benchmark return-series loading with canonical `date` / `benchmark_return` normalization
- CSV and Parquet corporate-actions loading with canonical `symbol` / `ex_date` / `action_type` / `split_ratio` / `cash_amount` normalization
- CSV and Parquet long-form fundamentals loading with canonical `symbol` / `period_end_date` / `release_date` / `metric_name` / `metric_value` normalization
- CSV and Parquet shares-outstanding loading with canonical `symbol` / `effective_date` / `shares_outstanding` normalization
- CSV and Parquet sector/industry classifications loading with canonical `symbol` / `effective_date` / `sector` / `industry` normalization
- CSV and Parquet index membership loading with canonical `symbol` / `effective_date` / `index_name` / `is_member` normalization
- CSV and Parquet borrow availability loading with canonical `symbol` / `effective_date` / `is_borrowable` / `borrow_fee_bps` normalization
- CSV and Parquet trading status loading with canonical `symbol` / `effective_date` / `is_tradable` / `status_reason` normalization
- Optional config-driven split-adjusted OHLCV loading with explicit backward price/volume adjustment factors
- CSV and Parquet symbol metadata loading with canonical `symbol` / `listing_date` / `delisting_date` normalization
- Deterministic sorting by `symbol` and `date`
- Forward returns, rolling volatility, and rolling average volume
- Optional next-session-safe fundamentals joins into the research dataset with explicit metric selection
- Optional valuation-style fundamental-to-price ratios for explicitly selected PIT fundamentals
- Optional quality-style fundamental ratios for explicitly selected PIT numerator/denominator metric pairs
- Optional growth-style period-over-period changes for explicitly selected PIT fundamentals
- Optional stability-style balance-sheet ratios for explicitly selected PIT numerator/denominator metric pairs
- Optional effective-date-safe sector/industry classification joins into the research dataset with explicit field selection
- Optional effective-date-safe index membership joins into the research dataset with explicit index selection
- Optional effective-date-safe borrow availability joins into the research dataset with explicit field selection
- Optional effective-date-safe trading status joins into the research dataset
- Optional effective-date-safe shares-outstanding joins that generate `shares_outstanding` and `market_cap`
- Optional trading calendar joins with fail-fast off-calendar date validation
- Optional trading-calendar validation for corporate-action `ex_date` values under `validate-data`
- Optional symbol metadata joins with fail-fast listing/delisting window validation
- Calendar-aware and metadata-aware listing-history counts for universe eligibility when those inputs are provided
- Lagged tradability-aware universe filtering with explicit eligibility diagnostics, including optional required effective-date-safe index membership and explicit trading status
- Optional within-date signal transforms with explicit `winsorize_quantile`, `clip_lower_bound` / `clip_upper_bound`, same-date numeric exposure residualization via `cross_sectional_residualize_columns`, same-date grouped de-meaning via `cross_sectional_neutralize_group_column`, and `cross_sectional_normalization` settings including `zscore`, `robust_zscore`, or `rank`; normalization can optionally be scoped within same-date groups such as `classification_sector`
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
- Optional explicit `position_cap_column` for row-level position caps, such as
  precomputed liquidity-aware weight caps
- Optional `group_column` plus `max_group_weight` to cap same-date exposure
  by an explicit dataset group such as `classification_sector` or
  `classification_industry`
- Optional shrink-only `factor_exposure_bounds` to cap target-weight net
  exposure to explicit numeric dataset columns
- Daily, weekly, and monthly rebalancing
- Split commission/slippage costs with legacy `transaction_cost_bps` compatibility
- Optional turnover caps with target vs realized execution diagnostics

### Analytics and Visualization

- Cumulative return, annualized return, volatility, Sharpe, drawdown, hit rate
- Benchmark-relative return, tracking error, and information ratio
- Rolling beta and rolling correlation versus a benchmark
- Target-weight portfolio diversification metrics based on absolute weights,
  effective number of positions, and top-name concentration
- Target-weight group exposure diagnostics when a portfolio `group_column` is
  configured
- Target-weight numeric exposure diagnostics for explicitly configured dataset
  columns
- IC / Rank IC summaries plus trailing rolling IC and IC decay diagnostics
- Quantile bucket returns and top-bottom quantile spread diagnostics
- Signal coverage summary and coverage-through-time diagnostics
- Static PNG charts for NAV, drawdown, exposure/turnover, IC, cumulative IC, grouped diagnostics, coverage, quantile diagnostics, and benchmark risk
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
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

AlphaForge supports Python 3.10 and newer. CI runs the test suite on Python 3.10, 3.11, 3.12, 3.13, and 3.14; the committed `.python-version` selects Python 3.14 for local version managers that honor it.

For branch protection, require the stable GitHub Actions check named `required-checks` instead of individual matrix checks such as `test (3.10)`. The aggregate check fails unless the full Python matrix succeeds, which avoids stale required checks when the matrix changes.

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
- `configs/market_cap_grouped_diagnostics_example.toml`

The example datasets in `data/raw/` are deterministic and synthetic. They are intended for reproducibility and pipeline inspection, not for claims about tradable alpha.

`configs/market_cap_grouped_diagnostics_example.toml` demonstrates the optional shares-outstanding join, same-date market-cap buckets, and explicit grouped IC / grouped coverage diagnostics with `diagnostics.group_columns = ["market_cap_bucket"]`.

Example universe membership filter settings:

```toml
[universe]
required_membership_indexes = ["S&P 500"]
lag = 1

[memberships]
path = "memberships.csv"
effective_date_column = "effective_date"
index_column = "index_name"
is_member_column = "is_member"
```

Membership universe filters use the same effective-date-safe membership join as
dataset descriptors, then apply the configured `universe.lag` before deciding
eligibility. Missing lagged membership status and explicit non-membership both
exclude the row. This is a conservative tradability / universe membership
filter; it does not model constituent weights, intraday membership timing, or
survivorship bias by itself.

Example trading status universe filter settings:

```toml
[universe]
require_tradable = true
lag = 1

[trading_status]
path = "trading_status.csv"
effective_date_column = "effective_date"
is_tradable_column = "is_tradable"
status_reason_column = "status_reason"
```

Trading status filters use the effective-date-safe trading status join, then
apply `universe.lag` before deciding eligibility. Missing lagged status and
explicit `is_tradable = false` both exclude the row. This is a conservative
halt/suspension-style research filter; it does not model limit-up/limit-down,
intraday halt timing, order execution, or fill realism.

Example signal transform settings:

```toml
[signal]
name = "momentum"
lookback = 20
winsorize_quantile = 0.05
clip_lower_bound = -3.0
clip_upper_bound = 3.0
# Optional: residualize against same-date numeric exposures already present in the dataset.
# cross_sectional_residualize_columns = ["rolling_benchmark_beta_20d"]
# Optional: de-mean within same-date groups after classifications are attached.
# cross_sectional_neutralize_group_column = "classification_sector"
cross_sectional_normalization = "robust_zscore"
# Optional: normalize within same-date groups after classifications are attached.
# cross_sectional_group_column = "classification_sector"
```

These transforms are applied within each date only, after any lagged universe eligibility mask has already removed ineligible rows. Explicit clipping requires both numeric bounds and preserves missing values; it is a conservative score-bounding step, not an execution constraint or liquidity model. If `cross_sectional_residualize_columns` is set, the signal is residualized with a same-date OLS regression against finite numeric exposure columns; rows with missing exposures, underidentified date groups, or non-full-rank exposure matrices remain missing rather than being converted into artificial neutral scores. If `cross_sectional_neutralize_group_column` is set, the signal is de-meaned within `date` plus that group before optional normalization; single-name or missing-group rows remain missing. Robust z-score normalization uses same-date median and scaled median absolute deviation, returning missing values when the robust scale is not available. If `cross_sectional_group_column` is set, only the normalization step is grouped by `date` plus that column; winsorization, explicit clipping, and residualization remain date-wide. This residualization is a simple research transform for numeric exposures, not automatic categorical sector one-hot residualization, portfolio exposure control, or execution realism. The built-in transform steps are also exposed through a small registry that records accepted parameters, default output suffixes, and same-date timing metadata.

The built-in signal names are also exposed through a small factor-definition registry. Each definition records accepted parameters, default output-column naming, required columns, and close-anchored timing metadata. This is a reusable wrapper around the existing signal builders; it is not a factor DAG, cache, or composite-alpha engine.

Example portfolio group constraint settings:

```toml
[portfolio]
construction = "long_only"
top_n = 20
weighting = "score"
exposure = 1.0
max_position_weight = 0.08
# Optional: a precomputed maximum absolute weight column already present in the dataset.
position_cap_column = "liquidity_weight_cap"
# Requires this column to already exist in the dataset.
group_column = "classification_sector"
max_group_weight = 0.30
# Requires this numeric column to already exist in the dataset.
factor_exposure_bounds = [
  { column = "rolling_benchmark_beta_20d", min = -0.10, max = 0.10 },
]
```

Global and row-level position caps are applied before group constraints, within
each rebalance date. `position_cap_column` must already exist in the dataset and
must contain nonnegative maximum absolute weights in portfolio-weight units; it
can be produced by a liquidity policy outside AlphaForge, but the framework does
not infer AUM, market impact, or executable shares from raw volume. Missing cap
values are treated as zero for selected names. For long-short portfolios,
position and group caps are applied independently to long and short side absolute
exposure. Missing or blank group labels are zero-weighted rather than assigned
to a fallback bucket, and the remaining exposure is left as cash or unused side
exposure rather than re-optimized into other groups.

Factor exposure bounds are applied after position and group caps to the combined
same-date target book. Each configured bound uses the net weighted exposure
`sum(portfolio_weight * exposure_column)`. If a bound is violated, AlphaForge
only shrinks the weights contributing in the violating direction toward zero; it
does not reallocate leftover exposure or solve an optimizer. Missing exposure
values for active selected names are zero-weighted because the bound cannot be
verified for those rows. Bounds must include zero, so this shrink-only method can
always fall back to cash or unused side exposure rather than invent offsetting
positions.

Example valuation-feature settings:

```toml
[dataset]
valuation_metrics = ["eps"]

[fundamentals]
path = "fundamentals.csv"
period_end_column = "period_end_date"
release_date_column = "release_date"
metric_name_column = "metric_name"
metric_value_column = "metric_value"
```

This dataset feature first attaches selected fundamentals with the existing next-session-safe release-date convention, then writes `valuation_<metric>_to_price = fundamental_<metric> / close`. It is intended for per-share or otherwise price-comparable metrics; AlphaForge does not infer market capitalization here.

Optional shares-outstanding files can be validated independently and, when explicitly enabled, used for market-cap features:

```toml
[dataset]
include_market_cap = true
market_cap_bucket_count = 3

[shares_outstanding]
path = "shares_outstanding.csv"
effective_date_column = "effective_date"
shares_outstanding_column = "shares_outstanding"
```

When `dataset.include_market_cap = true`, AlphaForge joins shares outstanding with the same effective-date convention as other reference data: an `effective_date` becomes active on the first market session not earlier than that date. The dataset then writes `shares_outstanding` and `market_cap = close * shares_outstanding`. Missing pre-effective rows remain missing. This feature does not automatically drive valuation ratios, portfolio constraints, or backtest behavior.

When `dataset.market_cap_bucket_count` is set, AlphaForge also writes `market_cap_bucket` as a same-date cross-sectional descriptor. Bucket `1` is the smallest market-cap bucket for that date. Dates with too few usable names, too few distinct positive market caps, or unstable duplicate quantile edges are left missing rather than forced into arbitrary buckets. The column is intended for explicit grouped diagnostics such as `diagnostics.group_columns = ["market_cap_bucket"]`, or as an explicit portfolio `group_column`; it does not change signal, portfolio, or backtest behavior unless referenced by configuration.

Example quality-feature settings:

```toml
[dataset]
quality_ratio_metrics = [["net_income", "total_assets"]]

[fundamentals]
path = "fundamentals.csv"
period_end_column = "period_end_date"
release_date_column = "release_date"
metric_name_column = "metric_name"
metric_value_column = "metric_value"
```

This dataset feature first attaches the selected numerator and denominator fundamentals with the existing next-session-safe release-date convention, then writes `quality_<numerator>_to_<denominator> = fundamental_<numerator> / fundamental_<denominator>`. Nonpositive denominators are treated as unavailable rather than forced into finite ratios.

Example growth-feature settings:

```toml
[dataset]
growth_metrics = ["revenue"]

[fundamentals]
path = "fundamentals.csv"
period_end_column = "period_end_date"
release_date_column = "release_date"
metric_name_column = "metric_name"
metric_value_column = "metric_value"
```

This dataset feature compares adjacent `period_end_date` observations for each selected metric and writes `growth_<metric> = current / prior - 1`. The growth value becomes available only on the current period's next-session-safe release date. Nonpositive prior values are treated as unavailable, and restatement lineage is not inferred.

Example balance-sheet stability-feature settings:

```toml
[dataset]
stability_ratio_metrics = [["total_debt", "total_assets"]]

[fundamentals]
path = "fundamentals.csv"
period_end_column = "period_end_date"
release_date_column = "release_date"
metric_name_column = "metric_name"
metric_value_column = "metric_value"
```

This dataset feature first attaches the selected numerator and denominator fundamentals with the existing next-session-safe release-date convention, then writes `stability_<numerator>_to_<denominator> = fundamental_<numerator> / fundamental_<denominator>`. Nonpositive denominators are treated as unavailable. AlphaForge does not infer whether a ratio is good or bad; the column is a timing-safe balance-sheet descriptor for downstream research.

Report artifacts also include `dataset_feature_metadata`, a JSON-friendly provenance plan for configured feature and label columns. Each entry records the output column, role, feature family, data source, input columns or metrics, timing convention, missing-data policy, and parameters. Reports and shared research-context metadata also include `signal_pipeline_metadata`, which records the configured factor definition, factor parameters, raw signal column, same-date transform steps, and final signal column. The `feature_cache_metadata` block adds a stable SHA-256 cache identity for the feature/signal plan and keeps future-return labels separate from reusable feature columns. This metadata does not change calculations or imply alpha quality.

The feature package also exposes a small materialized cache helper for future cache workflows. `write_research_feature_cache` writes `features.parquet` plus `manifest.json` from a precomputed frame and cache metadata. It writes only `date`, `symbol`, reusable feature columns, and configured signal columns; future-return labels are recorded as excluded labels and are never materialized. `load_research_feature_cache` validates the manifest, schema version, row count, and optional expected cache key before returning cached columns. This is a file-level helper, not a CLI cache engine, invalidation service, or dataset-versioning system.

Diagnostics can also compute trailing rolling IC statistics from the already computed per-date IC series:

```toml
[diagnostics]
rolling_ic_window = 20
```

Each rolling value uses only IC observations dated on or before the current date. Missing daily IC values are ignored, and the summary remains unavailable until the configured trailing window has enough valid IC observations.

Report artifacts also summarize IC decay across the configured dataset labels:

```toml
[dataset]
forward_horizons = [1, 5, 10]
```

The IC decay table reuses the same signal and IC method for each generated `forward_return_<horizon>d` column. Report artifacts also store a long-form horizon-by-horizon IC series and render an `ic_decay_series.png` chart. The primary `diagnostics.forward_return_column` still controls the single-horizon IC, quantile, coverage, and rolling IC sections.

Quantile diagnostics include mean forward returns by within-date signal bucket, top-minus-bottom spread through time, cumulative quantile mean forward-return paths, and a spread stability summary. The stability summary reports the mean top-minus-bottom spread, sample dispersion, mean/std stability ratio, sign consistency, and latest spread from label-based quantile buckets. These quantile views are research diagnostics; they are not portfolio backtests, execution simulations, or claims of tradable NAV.

Reports can also compute grouped IC diagnostics for explicit dataset columns:

```toml
[diagnostics]
group_columns = ["classification_sector"]
```

Grouped IC uses the configured `diagnostics.forward_return_column`, IC method, and minimum observation count, then computes same-date cross-sectional IC independently inside each non-missing group. The same `group_columns` setting also drives grouped coverage diagnostics for signal, forward-return label, and jointly usable rows by date and by group. When grouped diagnostics are configured, report chart bundles also include `grouped_ic_timeseries.png`, `grouped_ic_summary.png`, `grouped_coverage_timeseries.png`, and `grouped_coverage_summary.png`. Missing group values are excluded rather than assigned to a fallback bucket. These are diagnostic views of factor behavior and data availability by explicit dataset group column; they are not sector/style regression neutralization, automatic group-column inference, or portfolio exposure constraints.

Reports can also summarize generated target-weight exposure to explicit numeric
dataset columns:

```toml
[diagnostics]
exposure_columns = ["rolling_benchmark_beta_20d", "market_cap"]
```

Numeric exposure summaries use generated `portfolio_weight` targets, not
post-turnover effective holdings. For each configured column, reports show
absolute-weighted average exposure, net weighted exposure, and active weight
coverage with missing exposure values surfaced separately. Missing exposures are
not filled or inferred, and these diagnostics do not neutralize, constrain, or
optimize the portfolio.

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

Example dollar-volume-shock settings:

```toml
[dataset]
dollar_volume_shock_window = 20
```

This dataset feature uses same-day `log(close * volume)` minus the trailing mean of the prior `window` log dollar-volume observations. The rolling baseline excludes the current day, and zero-dollar-volume observations are treated as unavailable.

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

Example benchmark-residual-return settings:

```toml
[dataset]
benchmark_residual_return_window = 20

[benchmark]
path = "benchmark.csv"
name = "S&P 500"
return_column = "benchmark_return"
rolling_window = 20
```

This dataset feature requires exact benchmark/date alignment. It estimates a one-factor market-model alpha/beta from the prior `window` strategy and benchmark returns, then writes the same-day residual return. The fitted exposure excludes the current day, while the residual itself is anchored at the current close.

## Results and Artifacts

Representative report outputs are documented in [RESULTS.md](RESULTS.md).

The current artifact design is intentionally lightweight:

- `results.csv` stores tabular outputs
- `report.txt` stores the plain-text report
- `metadata.json` stores a workflow snapshot, compact summaries, feature provenance, signal pipeline metadata, and cache identity metadata
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
551 passed
```

## Limitations

- Daily data only; no intraday timestamps or intraday execution modeling
- No market impact, borrow cost, queue position, or order book simulation
- No optimizer-based portfolio construction, benchmark-relative exposure
  constraints, or factor-neutral portfolio optimization; row-level position caps
  require an explicit precomputed cap column and do not infer execution capacity;
  factor exposure bounds are shrink-only caps on explicit numeric columns, not a
  general optimizer or beta-neutral portfolio construction engine
- Benchmark analysis is based on date-only simple return series, not constituent-level attribution
- Trading calendar support currently uses explicit date-only session lists, not multi-exchange or intraday session engines
- Corporate actions currently support split-adjusted OHLCV plus split/cash-dividend event contracts; cash dividends are still not applied to total-return or dividend-adjusted price series
- Fundamentals currently support a long-form release-date-aware contract plus next-session-safe dataset joins, simple fundamental-to-price valuation ratios, explicit numerator/denominator quality ratios, adjacent-period growth rates, and explicit balance-sheet stability ratios, but still do not model release-time-of-day, restatement lineage, shares-outstanding-aware valuation, or broader point-in-time reference joins
- Shares outstanding currently supports an effective-date data contract, `validate-data` summary, optional research-dataset `shares_outstanding` / `market_cap` columns, and optional same-date `market_cap_bucket` descriptors; it does not yet drive valuation ratios and only affects portfolio constraints when a generated column is explicitly configured as `portfolio.group_column`
- Classifications currently support only effective-date-safe sector/industry histories; they do not yet cover more complex classification lineage
- Borrow availability currently supports only effective-date-safe borrowable/fee histories; it does not yet drive short-sale constraints, borrow costs, or richer securities-financing workflows
- Memberships currently support effective-date-safe index membership histories and optional lagged universe eligibility filters; they do not yet model constituent weights, intraday membership timing, survivorship bias control, or broader reference-data lineage
- Trading status currently supports effective-date-safe tradable/not-tradable histories and optional lagged universe eligibility filters; it does not model intraday halt timing, limit-up/limit-down rules, partial-session trading, order execution, or fill realism
- Grouped IC diagnostics currently support explicitly configured dataset group columns such as `classification_sector`; they do not infer sector fields automatically and do not implement style regression or exposure attribution
- Portfolio group exposure diagnostics summarize target weights by explicit
  group column; they do not infer sectors, optimize exposures, or model
  benchmark-relative active risk
- Numeric exposure diagnostics summarize generated target weights for explicit
  numeric columns; they do not impose factor bounds, neutralize exposures, or
  infer a style model
- Portfolio diversification metrics summarize generated target weights; they do
  not optimize the portfolio, infer benchmark-relative concentration, or model
  post-turnover effective holdings
- Cross-sectional signal transforms currently cover within-date winsorization, explicit numeric clipping, numeric exposure residualization, grouped de-meaning, z-score, robust z-score, and rank normalization; they do not yet cover automatic categorical one-hot residualization, factor-neutral portfolio optimization, or multi-step robust scaling stacks
- Dataset-level rolling statistics currently cover average true range, normalized average true range, Amihud illiquidity, dollar volume shock, dollar volume z-score, volume shock, relative volume, relative dollar volume, Garman-Klass volatility, Parkinson volatility, Rogers-Satchell volatility, Yang-Zhang volatility, daily-return-based realized volatility families, trailing skew/kurtosis, exact-date-aligned trailing beta/correlation versus a single benchmark, and benchmark-residualized returns; they do not yet cover richer range-based estimators, intraday volatility estimators, multi-benchmark features, or broader residualization pipelines
- Symbol metadata currently covers symbol-level listing/delisting dates only, not identifier-history workflows
- Visual outputs are static PNG/HTML artifacts, not interactive dashboards
- Feature provenance, factor definitions, transform definitions, signal pipeline metadata, and feature cache helpers are metadata/lightweight registry and file-cache layers, not a full factor DAG, CLI cache engine, invalidation service, or dataset versioning system
- Artifact tracking remains intentionally file-based rather than database-backed
