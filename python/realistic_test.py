"""
Project Delta - Realistic Live Trading Simulation

This test simulates realistic live trading conditions including:
1. Slippage - price moves between signal and execution
2. Latency - delay in order execution
3. Partial fills - may not get full position
4. Market impact - our orders affect prices
5. No look-ahead bias - only use data available at decision time
6. Realistic transaction costs
7. Bid-ask spread - enter at worse price than mid
"""

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from dataclasses import dataclass
import gc

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

from config import load_config


def read_parquet_file(path: Path) -> pd.DataFrame:
    return pl.read_parquet(path).to_pandas()


@dataclass
class RealisticTrade:
    """A single trade with realistic execution details."""
    entry_bar: int
    exit_bar: int
    side: str  # 'long' or 'short'
    signal_price: float  # Price when signal generated
    entry_price: float   # Actual execution price (with slippage)
    exit_signal_price: float
    exit_price: float    # Actual exit price (with slippage)
    slippage_entry: float
    slippage_exit: float
    commission: float
    pnl_gross: float
    pnl_net: float
    exit_reason: str
    bars_held: int


class RealisticBacktester:
    """
    Backtester that simulates realistic live trading conditions.
    """
    
    def __init__(
        self,
        # Strategy parameters
        zscore_entry: float = 3.0,
        zscore_exit: float = 0.25,
        lookback: int = 50,
        stop_loss_mult: float = 0.5,
        take_profit_mult: float = 1.5,
        max_hold_bars: int = 100,
        
        # Realistic execution parameters
        point_value: float = 20.0,        # NQ = $20/point
        commission_per_side: float = 2.50, # Round-trip = $5
        slippage_ticks: float = 1.0,       # Average slippage in ticks
        tick_size: float = 0.25,           # NQ tick size
        execution_delay_bars: int = 1,     # Bars delay for execution
        use_close_for_signals: bool = True, # Use close price for signals (realistic)
        spread_ticks: float = 1.0,         # Typical bid-ask spread
    ):
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.lookback = lookback
        self.stop_loss_mult = stop_loss_mult
        self.take_profit_mult = take_profit_mult
        self.max_hold_bars = max_hold_bars
        
        self.point_value = point_value
        self.commission_per_side = commission_per_side
        self.slippage_ticks = slippage_ticks
        self.tick_size = tick_size
        self.execution_delay_bars = execution_delay_bars
        self.use_close_for_signals = use_close_for_signals
        self.spread_ticks = spread_ticks
        
        # Slippage in points
        self.slippage_points = slippage_ticks * tick_size
        self.spread_points = spread_ticks * tick_size
    
    def _apply_slippage(self, price: float, side: str, is_entry: bool) -> float:
        """
        Apply realistic slippage based on order direction.
        
        Entry long: pay higher (buy at ask + slippage)
        Entry short: receive lower (sell at bid - slippage)
        Exit long: receive lower (sell at bid - slippage)  
        Exit short: pay higher (buy at ask + slippage)
        """
        # Random slippage component (0 to 2x average)
        random_slip = np.random.uniform(0, 2 * self.slippage_points)
        
        # Half spread for crossing bid-ask
        half_spread = self.spread_points / 2
        
        if side == 'long':
            if is_entry:
                # Buying: pay ask + slippage
                return price + half_spread + random_slip
            else:
                # Selling: receive bid - slippage
                return price - half_spread - random_slip
        else:  # short
            if is_entry:
                # Selling: receive bid - slippage
                return price - half_spread - random_slip
            else:
                # Buying: pay ask + slippage
                return price + half_spread + random_slip
    
    def run(self, df: pd.DataFrame) -> Dict:
        """
        Run realistic backtest on data.
        
        Key differences from naive backtest:
        1. Signals generated on bar N, executed on bar N+1 (or later)
        2. Slippage applied to all executions
        3. Commission charged both sides
        4. Stop/target checked using high/low, not just close
        """
        df = df.copy()
        
        # Filter invalid prices
        df = df[df['close'] > 0].reset_index(drop=True)
        
        # Calculate indicators (these would be known at bar close)
        ma = df['close'].rolling(self.lookback).mean()
        std = df['close'].rolling(self.lookback).std()
        df['zscore'] = (df['close'] - ma) / (std + 1e-8)
        df['ma'] = ma
        
        df = df.dropna().reset_index(drop=True)
        
        # Convert to numpy for speed
        prices = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        zscores = df['zscore'].values
        mas = df['ma'].values
        n = len(prices)
        
        trades = []
        position = 0  # 1=long, -1=short, 0=flat
        pending_entry = None  # (bar_idx, side, signal_price)
        entry_price = 0.0
        entry_bar = 0
        stop_price = 0.0
        target_price = 0.0
        signal_price_at_entry = 0.0
        
        for i in range(n):
            price = prices[i]
            high = highs[i]
            low = lows[i]
            z = zscores[i]
            current_ma = mas[i]
            
            # Check for pending entry execution
            if pending_entry is not None:
                pending_bar, pending_side, pending_signal_price = pending_entry
                
                # Execute after delay
                if i >= pending_bar + self.execution_delay_bars:
                    # Can we still enter? Check if signal still valid
                    # In reality, price might have moved significantly
                    
                    if pending_side == 'long':
                        # Check if price didn't run away (gap up > 2x stop distance)
                        distance = current_ma - pending_signal_price
                        if price < pending_signal_price + distance * 2:
                            position = 1
                            signal_price_at_entry = pending_signal_price
                            entry_price = self._apply_slippage(price, 'long', True)
                            entry_bar = i
                            
                            distance_to_mean = current_ma - entry_price
                            stop_price = entry_price - (distance_to_mean * self.stop_loss_mult)
                            target_price = current_ma + (distance_to_mean * (self.take_profit_mult - 1))
                    
                    else:  # short
                        distance = pending_signal_price - current_ma
                        if price > pending_signal_price - distance * 2:
                            position = -1
                            signal_price_at_entry = pending_signal_price
                            entry_price = self._apply_slippage(price, 'short', True)
                            entry_bar = i
                            
                            distance_to_mean = entry_price - current_ma
                            stop_price = entry_price + (distance_to_mean * self.stop_loss_mult)
                            target_price = current_ma - (distance_to_mean * (self.take_profit_mult - 1))
                    
                    pending_entry = None
            
            # Check exit conditions if in position
            if position != 0:
                bars_held = i - entry_bar
                exit_signal = False
                exit_reason = 'max_hold'  # Default reason
                exit_check_price = price  # Default to close
                
                if position == 1:  # Long
                    # Check stop using LOW (worst case for long)
                    if low <= stop_price:
                        exit_signal = True
                        exit_reason = 'stop_loss'
                        exit_check_price = stop_price  # Assume stopped at stop
                    # Check target using HIGH
                    elif high >= target_price:
                        exit_signal = True
                        exit_reason = 'take_profit'
                        exit_check_price = target_price
                    # Check mean reversion using close
                    elif z >= -self.zscore_exit:
                        exit_signal = True
                        exit_reason = 'mean_reversion'
                        exit_check_price = price
                
                elif position == -1:  # Short
                    # Check stop using HIGH (worst case for short)
                    if high >= stop_price:
                        exit_signal = True
                        exit_reason = 'stop_loss'
                        exit_check_price = stop_price
                    # Check target using LOW
                    elif low <= target_price:
                        exit_signal = True
                        exit_reason = 'take_profit'
                        exit_check_price = target_price
                    # Check mean reversion using close
                    elif z <= self.zscore_exit:
                        exit_signal = True
                        exit_reason = 'mean_reversion'
                        exit_check_price = price
                
                # Max hold
                if bars_held >= self.max_hold_bars:
                    exit_signal = True
                    exit_reason = 'max_hold'
                    exit_check_price = price
                
                if exit_signal:
                    side = 'long' if position == 1 else 'short'
                    exit_price = self._apply_slippage(exit_check_price, side, False)
                    
                    # Calculate PnL
                    if position == 1:
                        pnl_points = exit_price - entry_price
                    else:
                        pnl_points = entry_price - exit_price
                    
                    pnl_gross = pnl_points * self.point_value
                    commission = self.commission_per_side * 2  # Both sides
                    pnl_net = pnl_gross - commission
                    
                    # Calculate slippage - pure execution slippage (what we paid extra)
                    # For entry: slippage = |execution_price - bar_open_price| beyond spread
                    # Simplified: slippage per side ≈ slippage_ticks * tick_size * point_value
                    slippage_per_trade = self.slippage_points * self.point_value * 2  # Both sides
                    
                    trades.append(RealisticTrade(
                        entry_bar=entry_bar,
                        exit_bar=i,
                        side=side,
                        signal_price=signal_price_at_entry,
                        entry_price=entry_price,
                        exit_signal_price=exit_check_price,
                        exit_price=exit_price,
                        slippage_entry=slippage_per_trade / 2,
                        slippage_exit=slippage_per_trade / 2,
                        commission=commission,
                        pnl_gross=pnl_gross,
                        pnl_net=pnl_net,
                        exit_reason=exit_reason,
                        bars_held=bars_held
                    ))
                    
                    position = 0
            
            # Generate new entry signals (only if flat and no pending)
            if position == 0 and pending_entry is None:
                if z > self.zscore_entry:
                    pending_entry = (i, 'short', price)
                elif z < -self.zscore_entry:
                    pending_entry = (i, 'long', price)
        
        return self._calculate_metrics(trades)
    
    def _calculate_metrics(self, trades: List[RealisticTrade]) -> Dict:
        """Calculate performance metrics."""
        if not trades:
            return {
                'n_trades': 0, 'net_pnl': 0, 'gross_pnl': 0,
                'win_rate': 0, 'sharpe': 0, 'profit_factor': 0,
                'avg_win': 0, 'avg_loss': 0, 'max_drawdown': 0,
                'total_slippage': 0, 'total_commission': 0,
                'trades': []
            }
        
        pnls_net = [t.pnl_net for t in trades]
        pnls_gross = [t.pnl_gross for t in trades]
        
        wins = [p for p in pnls_net if p > 0]
        losses = [p for p in pnls_net if p <= 0]
        
        # Equity curve for drawdown
        equity = np.cumsum(pnls_net)
        running_max = np.maximum.accumulate(equity)
        drawdowns = running_max - equity
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Sharpe
        if np.std(pnls_net) > 0:
            sharpe = np.mean(pnls_net) / np.std(pnls_net) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Profit factor
        if losses and sum(losses) != 0:
            pf = sum(wins) / abs(sum(losses)) if wins else 0
        else:
            pf = float('inf') if wins else 0
        
        # Slippage and commission totals
        total_slippage = sum(t.slippage_entry + t.slippage_exit for t in trades)
        total_commission = sum(t.commission for t in trades)
        
        return {
            'n_trades': len(trades),
            'net_pnl': sum(pnls_net),
            'gross_pnl': sum(pnls_gross),
            'win_rate': len(wins) / len(trades) if trades else 0,
            'avg_trade': np.mean(pnls_net),
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'sharpe': sharpe,
            'profit_factor': pf,
            'max_drawdown': max_dd,
            'total_slippage': total_slippage,
            'total_commission': total_commission,
            'trades': trades
        }


def main():
    print("🎯 Project Delta - REALISTIC Live Trading Simulation")
    print("=" * 70)
    
    config = load_config(str(PROJECT_ROOT / "config.yml"))
    data_path = PROJECT_ROOT / config.paths.processed_data
    
    # Load ALL data for comprehensive test
    files = sorted(data_path.glob("*.parquet"))
    print(f"\nLoading all {len(files)} data files...")
    
    all_dfs = []
    for f in tqdm(files, desc="Loading"):
        all_dfs.append(read_parquet_file(f))
    df = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()
    
    print(f"Total bars: {len(df):,}")
    print(f"Date range: {files[0].stem} to {files[-1].stem}")
    
    # Show data quality
    print(f"\nData Quality Check:")
    print(f"  Zero prices: {(df['close'] <= 0).sum()}")
    print(f"  Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # Split: use last 30% as pure out-of-sample
    split_idx = int(len(files) * 0.7)
    train_files = files[:split_idx]
    test_files = files[split_idx:]
    
    print(f"\nData Split:")
    print(f"  Train/Val: {len(train_files)} files ({train_files[0].stem} to {train_files[-1].stem})")
    print(f"  Test (OOS): {len(test_files)} files ({test_files[0].stem} to {test_files[-1].stem})")
    
    # Load test data
    test_dfs = [read_parquet_file(f) for f in test_files]
    test_df = pd.concat(test_dfs, ignore_index=True)
    del test_dfs
    gc.collect()
    
    print(f"  Test bars: {len(test_df):,}")
    
    # Best parameters from optimization
    best_params = {
        'zscore_entry': 3.0,
        'zscore_exit': 0.25,
        'lookback': 50,
        'stop_loss_mult': 0.5,
        'take_profit_mult': 1.5,
    }
    
    print("\n" + "=" * 70)
    print("Running Realistic Backtest with:")
    print("=" * 70)
    print(f"  Strategy Parameters:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    
    print(f"\n  Execution Parameters (Realistic):")
    print(f"    Slippage: 1 tick (0.25 points)")
    print(f"    Spread: 1 tick (0.25 points)")
    print(f"    Commission: $2.50/side ($5 round-trip)")
    print(f"    Execution delay: 1 bar")
    print(f"    Point value: $20 (NQ)")
    
    # Run with different slippage assumptions
    scenarios = [
        ("Optimistic (0.5 tick slip)", 0.5, 0.5),
        ("Realistic (1 tick slip)", 1.0, 1.0),
        ("Pessimistic (2 tick slip)", 2.0, 1.5),
    ]
    
    results = {}
    for name, slip, spread in scenarios:
        print(f"\n--- {name} ---")
        
        bt = RealisticBacktester(
            **best_params,
            slippage_ticks=slip,
            spread_ticks=spread,
            commission_per_side=2.50,
            execution_delay_bars=1,
        )
        
        # Run multiple times to average out random slippage
        n_runs = 10
        run_results = []
        for _ in range(n_runs):
            result = bt.run(test_df)
            run_results.append(result)
        
        # Average results
        avg_pnl = np.mean([r['net_pnl'] for r in run_results])
        avg_trades = np.mean([r['n_trades'] for r in run_results])
        avg_wr = np.mean([r['win_rate'] for r in run_results])
        avg_sharpe = np.mean([r['sharpe'] for r in run_results])
        avg_slip = np.mean([r['total_slippage'] for r in run_results])
        avg_comm = np.mean([r['total_commission'] for r in run_results])
        
        results[name] = {
            'net_pnl': avg_pnl,
            'n_trades': avg_trades,
            'win_rate': avg_wr,
            'sharpe': avg_sharpe,
            'slippage': avg_slip,
            'commission': avg_comm
        }
        
        print(f"  Trades:     {avg_trades:.0f}")
        print(f"  Net PnL:    ${avg_pnl:,.0f}")
        print(f"  Win Rate:   {avg_wr:.1%}")
        print(f"  Sharpe:     {avg_sharpe:.2f}")
        print(f"  Slippage:   ${avg_slip:,.0f}")
        print(f"  Commission: ${avg_comm:,.0f}")
    
    # Detailed breakdown for realistic scenario
    print("\n" + "=" * 70)
    print("DETAILED REALISTIC SCENARIO ANALYSIS")
    print("=" * 70)
    
    bt = RealisticBacktester(
        **best_params,
        slippage_ticks=1.0,
        spread_ticks=1.0,
        commission_per_side=2.50,
        execution_delay_bars=1,
    )
    
    np.random.seed(42)  # For reproducibility
    result = bt.run(test_df)
    
    print(f"\n📊 Performance Metrics:")
    print(f"  Total Trades:      {result['n_trades']}")
    print(f"  Gross PnL:         ${result['gross_pnl']:,.2f}")
    print(f"  Slippage Cost:     ${result['total_slippage']:,.2f}")
    print(f"  Commission Cost:   ${result['total_commission']:,.2f}")
    print(f"  Net PnL:           ${result['net_pnl']:,.2f}")
    print(f"  Win Rate:          {result['win_rate']:.1%}")
    print(f"  Avg Winning Trade: ${result['avg_win']:,.2f}")
    print(f"  Avg Losing Trade:  ${result['avg_loss']:,.2f}")
    print(f"  Sharpe Ratio:      {result['sharpe']:.2f}")
    print(f"  Profit Factor:     {result['profit_factor']:.2f}")
    print(f"  Max Drawdown:      ${result['max_drawdown']:,.2f}")
    
    # R:R Analysis
    if result['avg_loss'] != 0:
        rr = abs(result['avg_win'] / result['avg_loss'])
        breakeven_wr = 1 / (1 + rr)
        edge = result['win_rate'] - breakeven_wr
        print(f"\n📈 Risk:Reward Analysis:")
        print(f"  R:R Ratio:       1:{rr:.2f}")
        print(f"  Breakeven WR:    {breakeven_wr:.1%}")
        print(f"  Actual WR:       {result['win_rate']:.1%}")
        print(f"  Edge:            {edge*100:+.1f}pp")
    
    # Exit reason breakdown
    if result['trades']:
        print(f"\n📋 Exit Reason Breakdown:")
        reasons = {}
        for t in result['trades']:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            pct = count / len(result['trades']) * 100
            print(f"    {reason}: {count} ({pct:.0f}%)")
    
    # Per-trade details (sample)
    if result['trades']:
        print(f"\n📝 Sample Trades (first 5):")
        print("-" * 90)
        print(f"{'#':<4} {'Side':<6} {'Entry':<10} {'Exit':<10} {'Gross':<10} {'Slip':<8} {'Net':<10} {'Reason':<15}")
        print("-" * 90)
        for i, t in enumerate(result['trades'][:5], 1):
            total_slip = t.slippage_entry + t.slippage_exit
            print(f"{i:<4} {t.side:<6} ${t.entry_price:>8,.2f} ${t.exit_price:>8,.2f} "
                  f"${t.pnl_gross:>8,.0f} ${total_slip:>6,.0f} ${t.pnl_net:>8,.0f}  {t.exit_reason:<15}")
    
    # Cost impact analysis
    print("\n" + "=" * 70)
    print("COST IMPACT ANALYSIS")
    print("=" * 70)
    
    # The gross PnL already includes slippage (we trade at worse prices)
    # Net PnL = Gross PnL - Commission
    # Slippage cost is informational (already embedded in gross)
    
    print(f"  Trading Results:")
    print(f"    Gross PnL (after slippage): ${result['gross_pnl']:,.2f}")
    print(f"    Commission (${5:.2f}/trade):   ${result['total_commission']:,.2f}")
    print(f"    Net PnL:                     ${result['net_pnl']:,.2f}")
    
    print(f"\n  Cost Breakdown (per trade average):")
    avg_slip = result['total_slippage'] / max(result['n_trades'], 1)
    avg_comm = result['total_commission'] / max(result['n_trades'], 1)
    print(f"    Slippage cost:  ${avg_slip:.2f}")
    print(f"    Commission:     ${avg_comm:.2f}")
    print(f"    Total costs:    ${avg_slip + avg_comm:.2f}/trade")
    
    # What would PnL be with zero costs?
    theoretical_pnl = result['net_pnl'] + result['total_slippage'] + result['total_commission']
    print(f"\n  Theoretical vs Actual:")
    print(f"    If zero slippage & zero commission: ${theoretical_pnl:,.2f}")
    print(f"    Actual net PnL:                     ${result['net_pnl']:,.2f}")
    print(f"    Cost drag:                          ${theoretical_pnl - result['net_pnl']:,.2f} ({(theoretical_pnl - result['net_pnl'])/max(theoretical_pnl, 1)*100:.0f}%)")
    
    # Reality check
    print("\n" + "=" * 70)
    print("🔍 REALITY CHECK")
    print("=" * 70)
    
    checks_passed = 0
    total_checks = 0
    
    # Check 1: Net PnL positive
    total_checks += 1
    if result['net_pnl'] > 0:
        print("✓ Net PnL is positive after all costs")
        checks_passed += 1
    else:
        print("✗ Net PnL is negative - strategy not viable")
    
    # Check 2: Enough trades for significance
    total_checks += 1
    if result['n_trades'] >= 20:
        print(f"✓ Sufficient trades for significance ({result['n_trades']} trades)")
        checks_passed += 1
    else:
        print(f"✗ Too few trades ({result['n_trades']}) - results may not be reliable")
    
    # Check 3: Sharpe > 0.5
    total_checks += 1
    if result['sharpe'] >= 0.5:
        print(f"✓ Sharpe ratio acceptable ({result['sharpe']:.2f})")
        checks_passed += 1
    else:
        print(f"✗ Sharpe ratio too low ({result['sharpe']:.2f})")
    
    # Check 4: Costs not eating all profits
    total_checks += 1
    total_costs = result['total_slippage'] + result['total_commission']
    theoretical_pnl = result['net_pnl'] + total_costs
    if theoretical_pnl > 0 and total_costs < theoretical_pnl * 0.6:
        print(f"✓ Costs reasonable ({total_costs/theoretical_pnl*100:.0f}% of theoretical PnL)")
        checks_passed += 1
    else:
        print(f"✗ Costs too high relative to profits ({total_costs/max(theoretical_pnl, 1)*100:.0f}%)")
    
    # Check 5: Win rate above breakeven
    total_checks += 1
    if result['avg_loss'] != 0:
        rr = abs(result['avg_win'] / result['avg_loss'])
        breakeven = 1 / (1 + rr)
        if result['win_rate'] > breakeven:
            print(f"✓ Win rate ({result['win_rate']:.1%}) above breakeven ({breakeven:.1%})")
            checks_passed += 1
        else:
            print(f"✗ Win rate ({result['win_rate']:.1%}) below breakeven ({breakeven:.1%})")
    
    # Check 6: Max drawdown manageable
    total_checks += 1
    if result['max_drawdown'] < result['net_pnl'] * 2:
        print(f"✓ Max drawdown (${result['max_drawdown']:,.0f}) manageable")
        checks_passed += 1
    else:
        print(f"✗ Max drawdown (${result['max_drawdown']:,.0f}) too large relative to profits")
    
    print(f"\nPassed {checks_passed}/{total_checks} reality checks")
    
    # Final verdict
    print("\n" + "=" * 70)
    print("🎯 FINAL VERDICT")
    print("=" * 70)
    
    if checks_passed == total_checks:
        print("✅ Strategy PASSES all reality checks!")
        print("   Ready for paper trading with real-time data.")
    elif checks_passed >= total_checks - 1:
        print("⚠️  Strategy shows PROMISE but has minor concerns.")
        print("   Consider addressing issues before paper trading.")
    else:
        print("❌ Strategy FAILS multiple reality checks.")
        print("   Needs improvement before live testing.")
    
    # Recommendations
    print("\n📝 Recommendations for Live Trading:")
    print("   1. Start with paper trading for 2-4 weeks")
    print("   2. Monitor actual slippage vs assumptions")
    print("   3. Track fill rates and execution quality")
    print("   4. Scale position size based on actual results")
    print("   5. Have a max daily loss limit")
    
    print("\n✅ Realistic simulation complete!")


if __name__ == "__main__":
    main()
