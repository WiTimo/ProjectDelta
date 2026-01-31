"""
Project Delta - Live Trading Engine

Reads real-time market data from NinjaTrader indicator output file,
preprocesses data into volume bars, generates signals using mean reversion,
and executes trades via hotkeys.

NinjaTrader Setup:
- F11 = Enter Long
- F12 = Enter Short
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Deque
from collections import deque
from datetime import datetime
import time
import logging
import sys

# Keyboard module for hotkeys (optional, installed separately)
keyboard = None  # type: ignore
try:
    import keyboard  # type: ignore  # pip install keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

from config import load_config


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup comprehensive logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"live_trading_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("LiveTrading")
    logger.setLevel(logging.DEBUG)
    
    # File handler - detailed
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Console handler - important messages only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class VolumeBar:
    """A completed volume bar with OHLC and features."""
    timestamp_start: datetime
    timestamp_end: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    trade_count: int
    buy_volume: int = 0
    sell_volume: int = 0
    vwap: float = 0.0


@dataclass
class Trade:
    """Record of an executed trade."""
    entry_time: datetime
    entry_bar_idx: int
    side: str  # 'long' or 'short'
    entry_price: float
    stop_price: float
    target_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None


@dataclass
class BarBuilder:
    """Builds volume bars from incoming tick data."""
    target_volume: int
    
    # Current bar state
    current_open: float = 0.0
    current_high: float = 0.0
    current_low: float = float('inf')
    current_close: float = 0.0
    current_volume: int = 0
    current_trades: int = 0
    current_buy_volume: int = 0
    current_sell_volume: int = 0
    current_vwap_sum: float = 0.0
    bar_start_time: Optional[datetime] = None
    
    # Best bid/ask for trade classification
    best_bid: float = 0.0
    best_ask: float = 0.0
    
    def update_quote(self, data_type: str, price: float, volume: int):
        """Update bid/ask from L1 data."""
        if data_type == "Bid":
            self.best_bid = price
        elif data_type == "Ask":
            self.best_ask = price
    
    def add_trade(self, price: float, volume: int, timestamp: datetime) -> Optional[VolumeBar]:
        """
        Add a trade and return completed bar if target volume reached.
        """
        if price <= 0 or volume <= 0:
            return None
        
        # Initialize bar if empty
        if self.current_volume == 0:
            self.current_open = price
            self.current_high = price
            self.current_low = price
            self.bar_start_time = timestamp
        
        # Update OHLC
        self.current_high = max(self.current_high, price)
        self.current_low = min(self.current_low, price)
        self.current_close = price
        self.current_volume += volume
        self.current_trades += 1
        self.current_vwap_sum += price * volume
        
        # Classify trade direction
        if self.best_bid > 0 and self.best_ask > 0:
            mid = (self.best_bid + self.best_ask) / 2
            if price >= mid:
                self.current_buy_volume += volume
            else:
                self.current_sell_volume += volume
        
        # Check if bar complete
        if self.current_volume >= self.target_volume:
            return self._complete_bar(timestamp)
        
        return None
    
    def _complete_bar(self, end_time: datetime) -> VolumeBar:
        """Complete current bar and reset."""
        vwap = self.current_vwap_sum / self.current_volume if self.current_volume > 0 else self.current_close
        
        bar = VolumeBar(
            timestamp_start=self.bar_start_time or end_time,
            timestamp_end=end_time,
            open=self.current_open,
            high=self.current_high,
            low=self.current_low,
            close=self.current_close,
            volume=self.current_volume,
            trade_count=self.current_trades,
            buy_volume=self.current_buy_volume,
            sell_volume=self.current_sell_volume,
            vwap=vwap
        )
        
        # Reset for next bar
        self.current_open = 0.0
        self.current_high = 0.0
        self.current_low = float('inf')
        self.current_close = 0.0
        self.current_volume = 0
        self.current_trades = 0
        self.current_buy_volume = 0
        self.current_sell_volume = 0
        self.current_vwap_sum = 0.0
        self.bar_start_time = None
        
        return bar


# =============================================================================
# LIVE TRADING ENGINE
# =============================================================================

class LiveTradingEngine:
    """
    Main live trading engine.
    
    Reads data from file, builds volume bars, generates signals,
    and executes trades via hotkeys.
    """
    
    def __init__(
        self,
        data_file: Path,
        log_dir: Path,
        # Strategy params
        volume_bar_size: int = 500,
        zscore_entry: float = 3.0,
        zscore_exit: float = 0.5,
        lookback: int = 50,
        stop_loss_mult: float = 0.5,
        take_profit_mult: float = 1.5,
        max_hold_bars: int = 100,
        # Hotkeys
        long_hotkey: str = 'F11',
        short_hotkey: str = 'F12',
        # Risk management
        max_daily_trades: int = 20,
        max_daily_loss: float = 500.0,
        point_value: float = 20.0,
        # Execution
        cooldown_seconds: float = 5.0,  # Min time between trades
    ):
        self.data_file = Path(data_file)
        self.log_dir = Path(log_dir)
        
        # Strategy params
        self.volume_bar_size = volume_bar_size
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.lookback = lookback
        self.stop_loss_mult = stop_loss_mult
        self.take_profit_mult = take_profit_mult
        self.max_hold_bars = max_hold_bars
        
        # Hotkeys
        self.long_hotkey = long_hotkey
        self.short_hotkey = short_hotkey
        
        # Risk management
        self.max_daily_trades = max_daily_trades
        self.max_daily_loss = max_daily_loss
        self.point_value = point_value
        self.cooldown_seconds = cooldown_seconds
        
        # State
        self.bar_builder = BarBuilder(target_volume=volume_bar_size)
        self.bars: Deque[VolumeBar] = deque(maxlen=lookback + 10)
        self.position: Optional[str] = None  # 'long', 'short', or None
        self.current_trade: Optional[Trade] = None
        self.trades: List[Trade] = []
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.last_trade_time: Optional[datetime] = None
        self.last_file_position: int = 0
        self.running: bool = False
        
        # Setup logging
        self.logger = setup_logging(log_dir)
    
    def parse_line(self, line: str) -> Optional[Dict]:
        """
        Parse a line of NinjaTrader market data.
        
        Expected formats:
        L1: DataType;Timestamp;Offset;Price;Volume
        L2: DataType;Timestamp;Offset;Operation;Position;Price;Volume
        """
        line = line.strip()
        if not line:
            return None
        
        parts = line.split(';')
        if len(parts) < 5:
            return None
        
        try:
            data_type_raw = parts[0]
            timestamp_str = parts[1]
            offset = int(parts[2]) if parts[2] else 0
            
            # Parse timestamp
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
            except ValueError:
                timestamp = datetime.now()
            
            # Determine L1 vs L2 by field count
            if len(parts) == 5:
                # L1 format
                price = float(parts[3].replace(',', '.'))
                volume = int(parts[4])
                
                # Map data type
                data_type_map = {
                    '0': 'Ask', '1': 'Bid', '2': 'Last',
                    'Ask': 'Ask', 'Bid': 'Bid', 'Last': 'Last'
                }
                data_type = data_type_map.get(data_type_raw, 'Unknown')
                
                return {
                    'type': 'L1',
                    'data_type': data_type,
                    'timestamp': timestamp,
                    'price': price,
                    'volume': volume
                }
            
            elif len(parts) >= 7:
                # L2 format
                operation = parts[3]
                position = int(parts[4])
                price = float(parts[5].replace(',', '.'))
                volume = int(parts[6])
                
                data_type_map = {'0': 'Ask', '1': 'Bid'}
                data_type = data_type_map.get(data_type_raw, 'Unknown')
                
                return {
                    'type': 'L2',
                    'data_type': data_type,
                    'operation': operation,
                    'position': position,
                    'timestamp': timestamp,
                    'price': price,
                    'volume': volume
                }
        
        except (ValueError, IndexError) as e:
            self.logger.debug(f"Parse error: {e} - Line: {line}")
            return None
        
        return None
    
    def process_tick(self, data: Dict) -> Optional[VolumeBar]:
        """Process a single tick and return completed bar if any."""
        data_type = data.get('data_type', '')
        price = data.get('price', 0)
        volume = data.get('volume', 0)
        timestamp = data.get('timestamp', datetime.now())
        
        # Update quotes
        if data_type in ['Bid', 'Ask']:
            self.bar_builder.update_quote(data_type, price, volume)
        
        # Process trades
        if data_type == 'Last' and price > 0 and volume > 0:
            return self.bar_builder.add_trade(price, volume, timestamp)
        
        return None
    
    def calculate_zscore(self) -> Optional[float]:
        """Calculate current Z-score from recent bars."""
        if len(self.bars) < self.lookback:
            return None
        
        closes = [b.close for b in list(self.bars)[-self.lookback:]]
        ma = np.mean(closes)
        std = np.std(closes)
        
        if std < 1e-8:
            return 0.0
        
        current_price = self.bars[-1].close
        return float((current_price - ma) / std)
    
    def check_exit_conditions(self, current_bar: VolumeBar) -> Optional[str]:
        """Check if current position should be exited."""
        if not self.current_trade or not self.position:
            return None
        
        price = current_bar.close
        high = current_bar.high
        low = current_bar.low
        
        bars_held = len(self.bars) - self.current_trade.entry_bar_idx
        zscore = self.calculate_zscore()
        
        if self.position == 'long':
            # Stop loss (check low)
            if low <= self.current_trade.stop_price:
                return 'stop_loss'
            # Take profit (check high)
            if high >= self.current_trade.target_price:
                return 'take_profit'
            # Mean reversion
            if zscore is not None and zscore >= -self.zscore_exit:
                return 'mean_reversion'
        
        elif self.position == 'short':
            # Stop loss (check high)
            if high >= self.current_trade.stop_price:
                return 'stop_loss'
            # Take profit (check low)
            if low <= self.current_trade.target_price:
                return 'take_profit'
            # Mean reversion
            if zscore is not None and zscore <= self.zscore_exit:
                return 'mean_reversion'
        
        # Max hold
        if bars_held >= self.max_hold_bars:
            return 'max_hold'
        
        return None
    
    def execute_entry(self, side: str, price: float, bar_idx: int):
        """Execute trade entry via hotkey."""
        now = datetime.now()
        
        # Cooldown check
        if self.last_trade_time:
            elapsed = (now - self.last_trade_time).total_seconds()
            if elapsed < self.cooldown_seconds:
                self.logger.debug(f"Cooldown active, {self.cooldown_seconds - elapsed:.1f}s remaining")
                return
        
        # Daily limits check
        if self.daily_trades >= self.max_daily_trades:
            self.logger.warning(f"Daily trade limit reached ({self.max_daily_trades})")
            return
        
        if self.daily_pnl <= -self.max_daily_loss:
            self.logger.warning(f"Daily loss limit reached (${self.max_daily_loss})")
            return
        
        # Calculate stops
        closes = [b.close for b in list(self.bars)[-self.lookback:]]
        ma = np.mean(closes)
        entry_distance = abs(price - ma)
        
        if side == 'long':
            stop_price = price - entry_distance * self.stop_loss_mult
            target_price = price + entry_distance * self.take_profit_mult
            hotkey = self.long_hotkey
        else:
            stop_price = price + entry_distance * self.stop_loss_mult
            target_price = price - entry_distance * self.take_profit_mult
            hotkey = self.short_hotkey
        
        # Send hotkey
        self.logger.info(f"🚀 EXECUTING {side.upper()} @ {price:.2f} | Stop: {stop_price:.2f} | Target: {target_price:.2f}")
        
        try:
            if keyboard is not None:
                keyboard.press_and_release(hotkey)
                self.logger.debug(f"Sent hotkey: {hotkey}")
            else:
                self.logger.error("Keyboard module not available")
                return
        except Exception as e:
            self.logger.error(f"Failed to send hotkey: {e}")
            return
        
        # Record trade
        self.current_trade = Trade(
            entry_time=now,
            entry_bar_idx=bar_idx,
            side=side,
            entry_price=price,
            stop_price=float(stop_price),
            target_price=float(target_price)
        )
        self.position = side
        self.last_trade_time = now
        self.daily_trades += 1
    
    def execute_exit(self, reason: str, price: float):
        """Execute trade exit."""
        if not self.current_trade or not self.position:
            return
        
        now = datetime.now()
        
        # Calculate PnL
        if self.position == 'long':
            pnl = (price - self.current_trade.entry_price) * self.point_value
        else:
            pnl = (self.current_trade.entry_price - price) * self.point_value
        
        # Log exit
        self.logger.info(f"🏁 EXIT {self.position.upper()} @ {price:.2f} | Reason: {reason} | PnL: ${pnl:+.2f}")
        
        # Update trade record
        self.current_trade.exit_time = now
        self.current_trade.exit_price = price
        self.current_trade.exit_reason = reason
        self.current_trade.pnl = pnl
        
        self.trades.append(self.current_trade)
        self.daily_pnl += pnl
        
        # Reset position
        self.current_trade = None
        self.position = None
        
        # Log daily stats
        self.logger.info(f"📊 Daily: {self.daily_trades} trades, ${self.daily_pnl:+.2f} PnL")
    
    def on_bar_complete(self, bar: VolumeBar):
        """Handle completed volume bar."""
        bar_idx = len(self.bars)
        self.bars.append(bar)
        
        self.logger.debug(
            f"Bar {bar_idx}: O={bar.open:.2f} H={bar.high:.2f} "
            f"L={bar.low:.2f} C={bar.close:.2f} V={bar.volume}"
        )
        
        # Need enough history
        if len(self.bars) < self.lookback:
            self.logger.debug(f"Building history: {len(self.bars)}/{self.lookback} bars")
            return
        
        zscore = self.calculate_zscore()
        if zscore is None:
            return
        
        self.logger.debug(f"Z-score: {zscore:.2f}")
        
        # Check exit conditions if in position
        if self.position and self.current_trade:
            exit_reason = self.check_exit_conditions(bar)
            if exit_reason:
                # Determine exit price based on reason
                if exit_reason == 'stop_loss':
                    exit_price = self.current_trade.stop_price
                elif exit_reason == 'take_profit':
                    exit_price = self.current_trade.target_price
                else:
                    exit_price = bar.close
                
                self.execute_exit(exit_reason, exit_price)
        
        # Check entry conditions if flat
        if not self.position:
            if zscore < -self.zscore_entry:
                # Long signal
                self.logger.info(f"📈 LONG SIGNAL: Z-score={zscore:.2f}")
                self.execute_entry('long', bar.close, bar_idx)
            
            elif zscore > self.zscore_entry:
                # Short signal
                self.logger.info(f"📉 SHORT SIGNAL: Z-score={zscore:.2f}")
                self.execute_entry('short', bar.close, bar_idx)
    
    def read_new_lines(self) -> List[str]:
        """Read new lines from data file."""
        if not self.data_file.exists():
            return []
        
        try:
            with open(self.data_file, 'r') as f:
                f.seek(self.last_file_position)
                new_lines = f.readlines()
                self.last_file_position = f.tell()
                return new_lines
        except Exception as e:
            self.logger.error(f"Error reading file: {e}")
            return []
    
    def run(self):
        """Main run loop."""
        self.running = True
        self.logger.info("=" * 60)
        self.logger.info("🚀 LIVE TRADING ENGINE STARTED")
        self.logger.info("=" * 60)
        self.logger.info(f"Data file: {self.data_file}")
        self.logger.info(f"Volume bar size: {self.volume_bar_size}")
        self.logger.info(f"Z-score entry: {self.zscore_entry}")
        self.logger.info(f"Stop loss mult: {self.stop_loss_mult}")
        self.logger.info(f"Take profit mult: {self.take_profit_mult}")
        self.logger.info(f"Long hotkey: {self.long_hotkey}")
        self.logger.info(f"Short hotkey: {self.short_hotkey}")
        self.logger.info("=" * 60)
        self.logger.info("Waiting for data... Press Ctrl+C to stop")
        
        try:
            while self.running:
                new_lines = self.read_new_lines()
                
                for line in new_lines:
                    data = self.parse_line(line)
                    if data:
                        completed_bar = self.process_tick(data)
                        if completed_bar:
                            self.on_bar_complete(completed_bar)
                
                # Small sleep to avoid busy waiting
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            self.logger.info("\n⛔ Shutdown requested by user")
        
        finally:
            self.stop()
    
    def stop(self):
        """Stop the engine and save logs."""
        self.running = False
        
        self.logger.info("=" * 60)
        self.logger.info("📊 SESSION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total trades: {len(self.trades)}")
        self.logger.info(f"Daily trades: {self.daily_trades}")
        self.logger.info(f"Daily PnL: ${self.daily_pnl:+.2f}")
        
        if self.trades:
            wins = [t for t in self.trades if t.pnl and t.pnl > 0]
            losses = [t for t in self.trades if t.pnl and t.pnl <= 0]
            win_rate = len(wins) / len(self.trades) * 100
            
            self.logger.info(f"Win rate: {win_rate:.1f}%")
            self.logger.info(f"Wins: {len(wins)}, Losses: {len(losses)}")
            
            if wins:
                win_pnls = [t.pnl for t in wins if t.pnl is not None]
                avg_win = float(np.mean(win_pnls)) if win_pnls else 0.0
                self.logger.info(f"Avg win: ${avg_win:.2f}")
            if losses:
                loss_pnls = [t.pnl for t in losses if t.pnl is not None]
                avg_loss = float(np.mean(loss_pnls)) if loss_pnls else 0.0
                self.logger.info(f"Avg loss: ${avg_loss:.2f}")
        
        # Save trade log
        self._save_trade_log()
        
        self.logger.info("=" * 60)
        self.logger.info("👋 Engine stopped")
    
    def _save_trade_log(self):
        """Save detailed trade log to CSV."""
        if not self.trades:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trade_file = self.log_dir / f"trades_{timestamp}.csv"
        
        rows = []
        for t in self.trades:
            rows.append({
                'entry_time': t.entry_time.isoformat() if t.entry_time else '',
                'exit_time': t.exit_time.isoformat() if t.exit_time else '',
                'side': t.side,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price or 0,
                'stop_price': t.stop_price,
                'target_price': t.target_price,
                'exit_reason': t.exit_reason or '',
                'pnl': t.pnl or 0
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(trade_file, index=False)
        self.logger.info(f"Trade log saved: {trade_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("🎯 Project Delta - Live Trading Engine")
    print("=" * 60)
    
    # Load config
    config = load_config(str(PROJECT_ROOT / "config.yml"))
    
    # Default paths
    data_file = PROJECT_ROOT / "data" / "live" / "market_data.txt"
    log_dir = PROJECT_ROOT / "logs"
    
    # Check if keyboard module is available
    if not KEYBOARD_AVAILABLE:
        print("❌ 'keyboard' module not installed!")
        print("   Run: pip install keyboard")
        print("   Note: Requires admin/root privileges on some systems")
        sys.exit(1)
    
    print(f"\nConfiguration:")
    print(f"  Data file:      {data_file}")
    print(f"  Log directory:  {log_dir}")
    print(f"  Volume bar:     {config.preprocessing.volume_bar_size}")
    print(f"  Z-score entry:  {config.strategy.zscore_entry}")
    print(f"  Stop loss:      {config.strategy.stop_loss_mult}x")
    print(f"  Take profit:    {config.strategy.take_profit_mult}x")
    print(f"  Long hotkey:    F11")
    print(f"  Short hotkey:   F12")
    
    # Create data directory if needed
    data_file.parent.mkdir(parents=True, exist_ok=True)
    
    if not data_file.exists():
        print(f"\n⚠️  Data file not found: {data_file}")
        print("   Create an empty file or configure NinjaTrader to write here.")
        # Create empty file
        data_file.touch()
        print(f"   Created empty file: {data_file}")
    
    print("\n" + "=" * 60)
    print("Starting live trading engine...")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    # Create and run engine
    engine = LiveTradingEngine(
        data_file=data_file,
        log_dir=log_dir,
        volume_bar_size=config.preprocessing.volume_bar_size,
        zscore_entry=config.strategy.zscore_entry,
        zscore_exit=config.strategy.zscore_exit,
        lookback=config.strategy.lookback,
        stop_loss_mult=config.strategy.stop_loss_mult,
        take_profit_mult=config.strategy.take_profit_mult,
        max_hold_bars=config.strategy.max_hold_bars,
        max_daily_trades=config.strategy.max_daily_trades,
        max_daily_loss=config.strategy.max_daily_loss,
        point_value=config.trading.point_value,
        long_hotkey='F11',
        short_hotkey='F12',
    )
    
    engine.run()


if __name__ == "__main__":
    main()
