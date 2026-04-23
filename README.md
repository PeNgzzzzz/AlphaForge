# AlphaForge

[![CI](https://github.com/PeNgzzzzz/AlphaForge/actions/workflows/ci.yml/badge.svg)](https://github.com/PeNgzzzzz/AlphaForge/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/PeNgzzzzz/AlphaForge)](LICENSE)

AlphaForge is a modular quant research workbench for daily cross-sectional equity research.

The project is built to be technically conservative, reproducible, and easy to explain. The emphasis is on timing correctness, transparent assumptions, inspectable backtests, and lightweight research tooling rather than exaggerated institutional realism.

- Representative outputs and charts: [RESULTS.md](RESULTS.md)

## What AlphaForge Covers

- Daily OHLCV data validation with explicit schema, duplicate checks, and conservative integrity rules.
- Research dataset construction with close-anchored features and forward-return labels.
- Optional lagged universe filters for price, rolling volume, rolling dollar volume, and listing history.
- Reusable price signals: momentum, mean reversion, and trend.
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
- Deterministic sorting by `symbol` and `date`
- Forward returns, rolling volatility, and rolling average volume
- Lagged tradability-aware universe filtering with explicit eligibility diagnostics

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
139 passed
```

## Limitations

- Daily data only; no intraday timestamps or intraday execution modeling
- No market impact, borrow cost, queue position, or order book simulation
- No optimizer-based portfolio construction or richer exposure constraints
- Benchmark analysis is based on dated return series, not constituent-level attribution
- Visual outputs are static PNG/HTML artifacts, not interactive dashboards
- Artifact tracking remains intentionally file-based rather than database-backed
