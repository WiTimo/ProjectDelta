# Project Delta - NQ Futures Mean Reversion Trading System

A complete day trading system for NQ (Nasdaq-100 E-mini futures) using microstructure features and mean reversion strategy.

## Overview

Project Delta processes NinjaTrader L1/L2 tick data into volume bars with order flow features, then applies a mean reversion strategy that trades when prices deviate significantly from their moving average.

### Key Features
- **High-performance Rust preprocessor** - Converts raw tick data to volume bars
- **Microstructure features** - Order Flow Imbalance (OFI), trade imbalance, depth ratios
- **Mean reversion strategy** - Z-score based entries with stop-loss/take-profit
- **Realistic backtesting** - Includes slippage, spread, commission, and execution delay
- **Walk-forward testing** - Out-of-sample validation across multiple time windows

### Strategy Performance (Realistic Backtest)
- **Net PnL**: $4,303 (36 trading days)
- **Win Rate**: 46.5%
- **Sharpe Ratio**: 2.22
- **Profit Factor**: 1.28
- **Risk/Reward**: 1:1.02

---

## Project Structure

```
ProjectDelta/
├── rust_preprocessor/      # Rust data preprocessing
│   ├── src/
│   │   ├── main.rs         # Entry point
│   │   ├── config.rs       # Configuration loading
│   │   ├── parser.rs       # L1/L2 data parser
│   │   ├── volume_bar.rs   # Volume bar aggregation
│   │   ├── order_book.rs   # Limit order book reconstruction
│   │   ├── features.rs     # Feature calculation (OFI, etc.)
│   │   └── output.rs       # Parquet/CSV output
│   └── Cargo.toml
├── python/                 # Python backtesting
│   ├── config.py           # Configuration loading
│   ├── data_split.py       # Chronological train/test splits
│   ├── mean_reversion.py   # Core strategy implementation
│   ├── optimize_rr.py      # Risk/reward parameter optimization
│   ├── extended_test.py    # Walk-forward analysis
│   └── realistic_test.py   # Realistic live trading simulation
├── data/
│   ├── raw/                # NinjaTrader CSV exports (YYYYMMDD.csv)
│   └── processed/          # Volume bar parquet files (auto-generated)
├── config.yml              # Shared configuration
└── requirements.txt        # Python dependencies
```

---

## Installation

### Prerequisites
- **Rust** (1.70+): https://rustup.rs/
- **Python** (3.10+): https://python.org/

### Setup

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd ProjectDelta
   ```

2. **Build the Rust preprocessor**
   ```bash
   cd rust_preprocessor
   cargo build --release
   cd ..
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Step 1: Preprocess Raw Data

Convert NinjaTrader tick data to volume bars:

```bash
cd rust_preprocessor
cargo run --release
```

This reads `data/raw/*.csv` and outputs `data/processed/*.parquet`.

**Configuration** (config.yml):
```yaml
preprocessing:
  volume_bar_size: 500    # Contracts per bar
  max_levels: 10          # Order book depth levels
  
paths:
  raw_data: "data/raw"
  processed_data: "data/processed"
```

### Step 2: Run Backtests

**Basic Strategy Test**
```bash
cd python
python mean_reversion.py
```

**Walk-Forward Analysis** (multiple OOS periods)
```bash
python extended_test.py
```

**Realistic Live Simulation** (with slippage, spread, commission)
```bash
python realistic_test.py
```

**Parameter Optimization**
```bash
python optimize_rr.py
```

---

## Strategy Details

### Mean Reversion Logic

1. **Calculate Z-score**: How many standard deviations price is from its moving average
2. **Entry Signal**: 
   - Long when Z-score < -3.0 (price well below average)
   - Short when Z-score > +3.0 (price well above average)
3. **Exit Conditions** (first hit wins):
   - Stop-loss: 0.5× entry distance from mean
   - Take-profit: 1.5× entry distance from mean
   - Mean reversion: Z-score crosses ±0.5
   - Timeout: 100 bars max hold

### Optimized Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Z-score Entry | 3.0 | Entry threshold (std devs) |
| Lookback | 50 | MA lookback period (bars) |
| Stop-loss | 0.5× | Multiple of entry distance |
| Take-profit | 1.5× | Multiple of entry distance |

### Features Used
- **Order Flow Imbalance (OFI)**: Measures buying vs selling pressure
- **Trade Imbalance**: Ratio of buy vs sell volume
- **Depth Ratio**: Bid depth vs ask depth
- **Spread**: Bid-ask spread statistics
- **VWAP Deviation**: Price deviation from volume-weighted average

---

## Live Trading Setup

### Requirements
1. NinjaTrader 8 with Market Depth data subscription
2. Real-time data feed (CME NQ futures)
3. Python environment with dependencies installed
4. `keyboard` Python package (requires admin privileges)

### NinjaTrader Indicator Setup

Create a NinjaTrader indicator that writes live market data to a file:

1. **Configure the indicator** to write L1/L2 data to:
   ```
   data/live/market_data.txt
   ```

2. **Data format** (same as historical exports):
   ```
   L1: DataType;Timestamp;Offset;Price;Volume
   L2: DataType;Timestamp;Offset;Operation;Position;Price;Volume
   ```

3. **Setup hotkeys in NinjaTrader**:
   - **F11** = Enter Long (Buy Market with your ATM strategy)
   - **F12** = Enter Short (Sell Market with your ATM strategy)

### Running the Live Trading Engine

1. **Install dependencies**:
   ```bash
   pip install keyboard
   ```
   Note: `keyboard` requires admin/root privileges on Windows.

2. **Start NinjaTrader** and ensure the indicator is writing data.

3. **Run the live trading engine**:
   ```bash
   cd python
   python live_trading.py
   ```

4. **Monitor the logs**:
   - Console shows real-time signals and trades
   - Detailed logs saved to `logs/live_trading_YYYYMMDD_HHMMSS.log`
   - Trade history saved to `logs/trades_YYYYMMDD_HHMMSS.csv`

### How It Works

1. **File Watching**: Engine monitors `data/live/market_data.txt` for new lines
2. **Bar Building**: Aggregates ticks into volume bars (500 contracts)
3. **Signal Generation**: Calculates Z-score and generates signals
4. **Trade Execution**: Sends F11/F12 hotkey to NinjaTrader
5. **Position Management**: Tracks stops, targets, and exit conditions
6. **Logging**: Records everything for analysis

### Risk Management (Live Trading)
- **Position Size**: 1 NQ contract = $20/point
- **Max Daily Loss**: $500 (stop trading for day)
- **Max Trades/Day**: 20
- **Cooldown**: 5 seconds between trades
- **Session Hours**: RTH only (9:30 AM - 4:00 PM ET)

### Stopping the Engine

Press `Ctrl+C` to gracefully stop. The engine will:
- Log session summary
- Save trade history to CSV
- Display final PnL and statistics

---

## Data Format

### Input: NinjaTrader CSV
```
L1: DataType;Timestamp;Offset;Price;Volume
L2: DataType;Timestamp;Offset;Operation;Position;Price;Volume
```

### Output: Volume Bar Parquet
| Column | Type | Description |
|--------|------|-------------|
| bar_index | u64 | Sequential bar number |
| timestamp_start | i64 | Bar start time (ns) |
| timestamp_end | i64 | Bar end time (ns) |
| open, high, low, close | f64 | OHLC prices |
| volume | u64 | Total volume (= bar_size) |
| trade_count | u32 | Number of trades |
| ofi_level1 | f64 | Level 1 order flow imbalance |
| ofi_aggregate | f64 | Aggregate OFI |
| buy_volume, sell_volume | u64 | Directional volume |
| trade_imbalance | f64 | (buy-sell)/(buy+sell) |
| depth_ratio | f64 | Bid depth / Ask depth |
| spread_avg, spread_max | f64 | Spread statistics |
| vwap, vwap_deviation | f64 | VWAP and deviation |

---

## Testing

### Validate Data Quality
```bash
cd rust_preprocessor
cargo run --release 2>&1 | grep -i "filter\|invalid"
```
The preprocessor automatically filters:
- Zero/negative prices
- Invalid OHLC relationships
- Bars with no trades

### Run All Tests
```bash
# Rust tests
cd rust_preprocessor && cargo test

# Python: Basic backtest
cd python && python mean_reversion.py

# Python: Realistic simulation
python realistic_test.py
```

---

## Configuration Reference

### config.yml
```yaml
preprocessing:
  volume_bar_size: 500      # Contracts per volume bar
  max_levels: 10            # Order book depth to track
  timezone: "US/Eastern"    # Timezone for session filtering

trading:
  point_value: 20.0         # NQ = $20 per point
  tick_size: 0.25           # NQ minimum tick
  commission_per_trade: 5.0 # Round-trip commission

strategy:
  zscore_entry: 3.0         # Entry threshold
  zscore_exit: 0.5          # Mean reversion exit
  lookback: 50              # MA lookback bars
  max_hold_bars: 100        # Maximum hold time
  stop_loss_mult: 0.5       # SL as multiple of entry distance
  take_profit_mult: 1.5     # TP as multiple of entry distance

paths:
  raw_data: "data/raw"
  processed_data: "data/processed"
```

---

## Troubleshooting

### Preprocessor Issues
- **"No CSV files found"**: Check `data/raw/` contains NinjaTrader exports
- **"Parse error"**: Ensure CSV uses correct delimiter (`;`)
- **Zero bars output**: Raw data may have no trades (quotes only)

### Python Issues
- **"No parquet files"**: Run Rust preprocessor first
- **Import errors**: Run `pip install -r requirements.txt`
- **Memory errors**: Process fewer files or use smaller volume bars

### Performance Issues
- Use `--release` flag for Rust builds (10x faster)
- Reduce `max_levels` if not using L2 features
- Process files in batches if memory limited

---

## License

MIT License - See LICENSE file for details.

---

## Changelog

### v1.0.0 (2025-01)
- Initial release with mean reversion strategy
- Rust preprocessor for L1/L2 data
- Realistic backtesting with slippage/commission
- Walk-forward validation

