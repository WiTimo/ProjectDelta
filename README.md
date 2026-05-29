# Project Delta

Project Delta is an NQ futures market microstructure research system built around a rules-based mean-reversion strategy. It converts NinjaTrader L1/L2 tick data into volume bars, calculates order-flow features, and evaluates a z-score mean-reversion strategy with realistic backtest assumptions.

Unlike the earlier ML-focused projects, Project Delta is intentionally simpler and more interpretable: it tests whether transparent order-flow and price-deviation rules can produce useful behavior before adding more complex machine learning.

## Repository description

**Interpretable NQ futures mean-reversion research system with Rust volume-bar preprocessing, L1/L2 order-flow features, walk-forward testing, and realistic backtest simulation.**

## Project status

This is a research and backtesting project. It is not a production trading system and does not guarantee future trading performance. Any real-world trading use would require independent validation, paper testing, broker/exchange compliance checks, operational monitoring, and strict risk controls.

## What it does

- Parses NinjaTrader L1/L2 tick data.
- Reconstructs order book state up to configurable depth levels.
- Aggregates raw ticks into volume bars.
- Computes microstructure features such as OFI, trade imbalance, depth ratio, spread, and VWAP deviation.
- Applies a z-score mean-reversion strategy.
- Runs realistic backtests with slippage, spread, commission, and execution delay.
- Supports walk-forward evaluation across multiple time windows.
- Contains research-only tooling for studying signal behavior on streamed market-data files.

## Why this project matters

Project Delta demonstrates that good trading research does not always start with a neural network. It focuses on building a transparent baseline system with realistic assumptions:

- **Interpretable logic:** entries and exits are based on z-score deviation from a moving average.
- **Market microstructure features:** volume bars and order-flow statistics provide context beyond simple OHLC data.
- **Realistic evaluation:** spread, slippage, commission, and execution delay are included.
- **Walk-forward validation:** the strategy is tested across chronological out-of-sample windows.
- **Clear risk controls:** daily loss, maximum trade count, position size, and cooldown settings are explicit in configuration.

## High-level architecture

```text
NinjaTrader L1/L2 CSV data
        ↓
Rust preprocessor
        ↓
Volume bars + order-flow features
        ↓
Python strategy engine
        ↓
Mean-reversion backtests
        ↓
Walk-forward and realistic execution analysis
        ↓
Research-only streamed-data signal study
```

## Tech stack

| Area | Technology |
|---|---|
| Preprocessing | Rust |
| Backtesting | Python |
| Strategy | Z-score mean reversion |
| Data format | CSV input, Parquet/CSV processed output |
| Configuration | YAML |
| Data source | NinjaTrader L1/L2 tick data |

## Repository structure

```text
ProjectDelta/
├── rust_preprocessor/
│   ├── src/
│   │   ├── main.rs              # Preprocessor entrypoint
│   │   ├── config.rs            # YAML configuration loading
│   │   ├── parser.rs            # L1/L2 input parser
│   │   ├── volume_bar.rs        # Volume bar aggregation
│   │   ├── order_book.rs        # Limit order book reconstruction
│   │   ├── features.rs          # OFI and market microstructure features
│   │   └── output.rs            # Parquet/CSV output
│   └── Cargo.toml
├── python/
│   ├── config.py                # Configuration loading
│   ├── data_split.py            # Chronological split utilities
│   ├── mean_reversion.py        # Core strategy implementation
│   ├── optimize_rr.py           # Risk/reward parameter testing
│   ├── extended_test.py         # Walk-forward analysis
│   └── realistic_test.py        # Realistic execution simulation
├── data/
│   ├── raw/                     # NinjaTrader CSV exports
│   └── preprocessed/            # Generated volume bar data
├── config.yml                   # Shared configuration
├── requirements.txt             # Python dependencies
└── README.md
```

## Strategy overview

Project Delta uses a z-score mean-reversion strategy:

1. Build volume bars from raw market data.
2. Calculate a moving average over a configurable lookback window.
3. Measure how far price is from that average in standard deviations.
4. Mark long-side candidates when price is significantly below the mean.
5. Mark short-side candidates when price is significantly above the mean.
6. Exit in the simulation on stop-loss, take-profit, mean reversion, or timeout.

Default strategy parameters:

| Parameter | Value | Meaning |
|---|---:|---|
| Z-score entry | `3.0` | Candidate threshold in standard deviations |
| Z-score exit | `0.5` | Exit when price reverts near the mean |
| Lookback | `50` bars | Moving-average window |
| Stop-loss | `0.5x` | Multiple of entry distance from mean |
| Take-profit | `1.5x` | Multiple of entry distance from mean |
| Max hold | `100` bars | Timeout exit |

## Features

The Rust preprocessor produces features such as:

| Feature | Meaning |
|---|---|
| OHLC | Open, high, low, close of each volume bar |
| Volume | Total contracts in the bar |
| Trade count | Number of trades in the bar |
| OFI level 1 | Level-1 order-flow imbalance |
| OFI aggregate | Multi-level order-flow imbalance |
| Buy/sell volume | Directional trade volume |
| Trade imbalance | `(buy - sell) / (buy + sell)` |
| Depth ratio | Bid depth divided by ask depth |
| Spread statistics | Average and maximum spread |
| VWAP deviation | Price deviation from volume-weighted average |

## Configuration

Main config file:

```text
config.yml
```

Important sections:

```yaml
paths:
  raw_data: "data/raw"
  processed_data: "data/preprocessed"

preprocessing:
  volume_bar_size: 500
  max_levels: 10
  ofi_levels: 5

trading:
  point_value: 20.0
  tick_size: 0.25
  commission_per_trade: 5.0
  slippage_ticks: 1
  spread_ticks: 1

strategy:
  zscore_entry: 3.0
  zscore_exit: 0.5
  lookback: 50
  stop_loss_mult: 0.5
  take_profit_mult: 1.5
  max_hold_bars: 100
```

## Installation

### Prerequisites

- Rust 1.70+
- Python 3.10+
- NinjaTrader market-data exports

### Build the Rust preprocessor

```bash
cd rust_preprocessor
cargo build --release
cd ..
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 1. Add raw data

Place NinjaTrader CSV files in:

```text
data/raw/
```

### 2. Preprocess data

```bash
cd rust_preprocessor
cargo run --release
cd ..
```

Processed data is written to the configured processed-data directory.

### 3. Run the base strategy test

```bash
cd python
python mean_reversion.py
```

### 4. Run walk-forward analysis

```bash
python extended_test.py
```

### 5. Run realistic execution simulation

```bash
python realistic_test.py
```

### 6. Run parameter testing

```bash
python optimize_rr.py
```

## Backtesting assumptions

The realistic simulation includes:

- Commission
- Slippage
- Bid/ask spread
- Execution delay
- Maximum daily trades
- Maximum daily loss
- Session-hour filtering

This is important because raw strategy results without execution assumptions are usually too optimistic.

## Testing and validation

Run Rust tests:

```bash
cd rust_preprocessor
cargo test
```

Run core Python strategy checks:

```bash
cd python
python mean_reversion.py
python realistic_test.py
```

Validate preprocessing output by checking logs for filtered or invalid bars:

```bash
cd rust_preprocessor
cargo run --release 2>&1 | grep -i "filter\|invalid"
```

## Recruiter notes

Project Delta is a strong portfolio project because it shows rigorous baseline research: high-performance preprocessing, interpretable strategy logic, order-flow feature engineering, walk-forward testing, and realistic execution modeling. It also demonstrates good judgment by separating research results from production trading claims.
