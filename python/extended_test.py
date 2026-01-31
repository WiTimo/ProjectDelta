"""
Project Delta - Extended Out-of-Sample Testing

Uses walk-forward analysis to test strategy robustness over multiple periods.
This gives a better estimate of true out-of-sample performance.
"""

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import gc
from dataclasses import dataclass

# Ensure we're using project root for relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

from config import load_config
from mean_reversion import mean_reversion_backtest, read_parquet_file


@dataclass
class WalkForwardResult:
    """Result from a single walk-forward window."""
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_trades: int
    net_pnl: float
    win_rate: float
    sharpe: float
    profit_factor: float
    best_params: dict


def walk_forward_test(
    files: List[Path],
    train_window: int = 15,  # Number of files for training
    test_window: int = 5,    # Number of files for testing
    step_size: int = 5       # How many files to move forward each iteration
) -> List[WalkForwardResult]:
    """
    Walk-forward testing with reoptimization.
    
    For each window:
    1. Optimize parameters on train window
    2. Test on out-of-sample test window
    3. Move forward by step_size files
    """
    results = []
    
    n_files = len(files)
    start_idx = 0
    
    print(f"\n🔄 Walk-Forward Analysis")
    print(f"   Train window: {train_window} files")
    print(f"   Test window:  {test_window} files")
    print(f"   Step size:    {step_size} files")
    print(f"   Total files:  {n_files}")
    
    window_num = 0
    while start_idx + train_window + test_window <= n_files:
        window_num += 1
        train_end_idx = start_idx + train_window
        test_end_idx = train_end_idx + test_window
        
        train_files = files[start_idx:train_end_idx]
        test_files = files[train_end_idx:test_end_idx]
        
        train_start = train_files[0].stem
        train_end = train_files[-1].stem
        test_start = test_files[0].stem
        test_end = test_files[-1].stem
        
        print(f"\n--- Window {window_num}: Train {train_start}-{train_end}, Test {test_start}-{test_end} ---")
        
        # Load training data
        train_dfs = [read_parquet_file(f) for f in train_files]
        train_df = pd.concat(train_dfs, ignore_index=True)
        del train_dfs
        gc.collect()
        
        # Optimize parameters on training data
        best_params = None
        best_sharpe = -np.inf
        
        for zscore_entry in [2.0, 2.5, 3.0]:
            for lookback in [30, 50]:
                for stop_loss_mult in [0.5, 0.75, 1.0]:
                    for take_profit_mult in [1.5, 2.0, 2.5]:
                        if take_profit_mult <= stop_loss_mult:
                            continue
                        
                        result = mean_reversion_backtest(
                            train_df,
                            zscore_entry=zscore_entry,
                            zscore_exit=0.25,
                            lookback=lookback,
                            stop_loss_mult=stop_loss_mult,
                            take_profit_mult=take_profit_mult
                        )
                        
                        if result['n_trades'] >= 10 and result['sharpe'] > best_sharpe:
                            best_sharpe = result['sharpe']
                            best_params = {
                                'zscore_entry': zscore_entry,
                                'zscore_exit': 0.25,
                                'lookback': lookback,
                                'stop_loss_mult': stop_loss_mult,
                                'take_profit_mult': take_profit_mult
                            }
        
        del train_df
        gc.collect()
        
        if not best_params:
            best_params = {
                'zscore_entry': 2.5,
                'zscore_exit': 0.25,
                'lookback': 30,
                'stop_loss_mult': 0.75,
                'take_profit_mult': 1.5
            }
        
        # Test on out-of-sample data
        test_dfs = [read_parquet_file(f) for f in test_files]
        test_df = pd.concat(test_dfs, ignore_index=True)
        del test_dfs
        gc.collect()
        
        test_result = mean_reversion_backtest(test_df, **best_params)
        
        del test_df
        gc.collect()
        
        result = WalkForwardResult(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            n_trades=test_result['n_trades'],
            net_pnl=test_result['net_pnl'],
            win_rate=test_result['win_rate'],
            sharpe=test_result['sharpe'],
            profit_factor=test_result['profit_factor'],
            best_params=best_params
        )
        
        results.append(result)
        
        print(f"   Params: z={best_params['zscore_entry']}, lb={best_params['lookback']}, "
              f"sl={best_params['stop_loss_mult']}, tp={best_params['take_profit_mult']}")
        print(f"   OOS Results: {test_result['n_trades']} trades, "
              f"${test_result['net_pnl']:,.0f}, WR={test_result['win_rate']:.1%}, "
              f"Sharpe={test_result['sharpe']:.2f}")
        
        # Move forward
        start_idx += step_size
    
    return results


def main():
    print("📊 Project Delta - Extended Out-of-Sample Testing")
    print("=" * 60)
    
    config = load_config(str(PROJECT_ROOT / "config.yml"))
    data_path = PROJECT_ROOT / config.paths.processed_data
    
    # Get all files sorted chronologically
    files = sorted(data_path.glob("*.parquet"))
    print(f"\nTotal data files: {len(files)}")
    print(f"Date range: {files[0].stem} to {files[-1].stem}")
    
    # Run walk-forward analysis
    results = walk_forward_test(
        files,
        train_window=15,
        test_window=5,
        step_size=5
    )
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("📈 AGGREGATED OUT-OF-SAMPLE RESULTS")
    print("=" * 60)
    
    total_trades = sum(r.n_trades for r in results)
    total_pnl = sum(r.net_pnl for r in results)
    avg_win_rate = np.mean([r.win_rate for r in results if r.n_trades > 0])
    avg_sharpe = np.mean([r.sharpe for r in results if r.n_trades > 0])
    
    # Calculate overall Sharpe from combined PnL series
    all_pnls = [r.net_pnl for r in results]
    if len(all_pnls) > 1:
        overall_sharpe = np.mean(all_pnls) / (np.std(all_pnls) + 1e-8) * np.sqrt(len(all_pnls))
    else:
        overall_sharpe = avg_sharpe
    
    # Count winning periods
    winning_periods = sum(1 for r in results if r.net_pnl > 0)
    
    print(f"\nWalk-Forward Windows: {len(results)}")
    print(f"Winning Periods:      {winning_periods}/{len(results)} ({winning_periods/len(results)*100:.0f}%)")
    print(f"\nTotal Trades:         {total_trades}")
    print(f"Total PnL:            ${total_pnl:,.2f}")
    print(f"Avg Win Rate:         {avg_win_rate:.1%}")
    print(f"Avg Period Sharpe:    {avg_sharpe:.2f}")
    print(f"Overall Sharpe:       {overall_sharpe:.2f}")
    
    # Per-window breakdown
    print("\n📋 Per-Window Details:")
    print(f"{'Window':<8} {'Test Period':<20} {'Trades':<8} {'PnL':<12} {'WR':<8} {'Sharpe':<8}")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        period = f"{r.test_start}-{r.test_end}"
        print(f"{i:<8} {period:<20} {r.n_trades:<8} ${r.net_pnl:>9,.0f}  {r.win_rate:>6.1%}  {r.sharpe:>6.2f}")
    
    # Consistency analysis
    print("\n🔍 Consistency Analysis:")
    if len(results) >= 2:
        sharpe_std = np.std([r.sharpe for r in results])
        pnl_std = np.std([r.net_pnl for r in results])
        
        print(f"   Sharpe Std Dev:  {sharpe_std:.2f}")
        print(f"   PnL Std Dev:     ${pnl_std:,.0f}")
        
        if sharpe_std < 1.0:
            print("   ✓ Reasonably consistent Sharpe across periods")
        else:
            print("   ⚠ High variability in Sharpe across periods")
        
        # Check for regime changes
        if any(r.sharpe < 0 for r in results):
            losing_count = sum(1 for r in results if r.sharpe < 0)
            print(f"   ⚠ {losing_count} periods had negative Sharpe (possible regime change)")
    
    # Parameter stability
    print("\n📐 Parameter Stability:")
    param_counts = {}
    for r in results:
        key = (r.best_params['zscore_entry'], r.best_params['lookback'], 
               r.best_params['stop_loss_mult'], r.best_params['take_profit_mult'])
        param_counts[key] = param_counts.get(key, 0) + 1
    
    print(f"   Unique param combos used: {len(param_counts)}/{len(results)} windows")
    most_common = max(param_counts.items(), key=lambda x: x[1])
    print(f"   Most common: z={most_common[0][0]}, lb={most_common[0][1]}, "
          f"sl={most_common[0][2]}, tp={most_common[0][3]} ({most_common[1]} times)")
    
    # Final verdict
    print("\n" + "=" * 60)
    print("🎯 FINAL VERDICT")
    print("=" * 60)
    
    passing = 0
    checks = 0
    
    checks += 1
    if total_pnl > 0:
        print("✓ Total OOS PnL is positive")
        passing += 1
    else:
        print("✗ Total OOS PnL is negative")
    
    checks += 1
    if winning_periods / len(results) >= 0.5:
        print("✓ Majority of periods are profitable")
        passing += 1
    else:
        print("✗ Less than 50% of periods are profitable")
    
    checks += 1
    if avg_sharpe >= 0.5:
        print("✓ Average Sharpe >= 0.5")
        passing += 1
    else:
        print("✗ Average Sharpe < 0.5")
    
    checks += 1
    if avg_win_rate >= 0.5:
        print("✓ Average win rate >= 50%")
        passing += 1
    else:
        print("✗ Average win rate < 50%")
    
    print(f"\nPassed {passing}/{checks} checks")
    
    if passing == checks:
        print("\n🎉 Strategy appears ROBUST for live trading consideration")
    elif passing >= checks - 1:
        print("\n⚠️ Strategy shows PROMISE but needs more validation")
    else:
        print("\n❌ Strategy needs IMPROVEMENT before live trading")
    
    print("\n✅ Extended testing complete!")


if __name__ == "__main__":
    main()
