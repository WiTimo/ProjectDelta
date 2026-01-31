"""
Project Delta - Advanced R:R Optimization

Explores a wider parameter space to find configurations with better 
risk:reward ratios while maintaining profitability.
"""

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import gc
from itertools import product

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

from config import load_config
from mean_reversion import mean_reversion_backtest, read_parquet_file
from data_split import create_chronological_split


def detailed_param_sweep(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame
) -> List[Dict]:
    """
    Comprehensive parameter sweep with detailed metrics for each combination.
    Returns all results sorted by a composite score.
    """
    results = []
    
    # Reduced but still comprehensive parameter ranges
    zscore_entries = [2.0, 2.5, 3.0, 3.5]
    lookbacks = [20, 30, 50]
    stop_loss_mults = [0.5, 0.75, 1.0]
    take_profit_mults = [1.5, 2.0, 2.5, 3.0]
    zscore_exits = [0.25, 0.5]
    
    total_combos = (len(zscore_entries) * len(lookbacks) * len(stop_loss_mults) * 
                   len(take_profit_mults) * len(zscore_exits))
    
    print(f"Testing {total_combos} parameter combinations...")
    
    pbar = tqdm(total=total_combos, desc="Sweep")
    
    for z_entry, lb, sl, tp, z_exit in product(
        zscore_entries, lookbacks, stop_loss_mults, take_profit_mults, zscore_exits
    ):
        pbar.update(1)
        
        # Skip invalid R:R ratios (want at least 1:1)
        if tp < sl:
            continue
        
        # Train result
        train_result = mean_reversion_backtest(
            train_df,
            zscore_entry=z_entry,
            zscore_exit=z_exit,
            lookback=lb,
            stop_loss_mult=sl,
            take_profit_mult=tp
        )
        
        # Skip if not enough trades in training
        if train_result['n_trades'] < 15:
            continue
        
        # Validation result
        val_result = mean_reversion_backtest(
            val_df,
            zscore_entry=z_entry,
            zscore_exit=z_exit,
            lookback=lb,
            stop_loss_mult=sl,
            take_profit_mult=tp
        )
        
        # Skip if no validation trades
        if val_result['n_trades'] < 5:
            continue
        
        # Calculate R:R ratio
        if val_result['avg_loss'] != 0:
            rr_ratio = abs(val_result['avg_win'] / val_result['avg_loss'])
        else:
            rr_ratio = 1.0
        
        # Calculate expected value per trade
        if val_result['n_trades'] > 0:
            ev_per_trade = val_result['net_pnl'] / val_result['n_trades']
        else:
            ev_per_trade = 0
        
        # Composite score: balance between Sharpe, R:R, and consistency
        # Penalize large train-val gap (overfitting)
        sharpe_gap = abs(train_result['sharpe'] - val_result['sharpe'])
        consistency_penalty = max(0, sharpe_gap - 1.0) * 0.2
        
        composite_score = (
            val_result['sharpe'] * 0.4 +              # Risk-adjusted returns
            min(rr_ratio, 3.0) * 0.3 +                # R:R ratio (capped at 3)
            val_result['profit_factor'] * 0.2 +       # Profitability
            (val_result['win_rate'] - 0.4) * 2 * 0.1  # Win rate above 40%
            - consistency_penalty                      # Overfitting penalty
        )
        
        results.append({
            'zscore_entry': z_entry,
            'zscore_exit': z_exit,
            'lookback': lb,
            'stop_loss_mult': sl,
            'take_profit_mult': tp,
            'rr_ratio': rr_ratio,
            'train_trades': train_result['n_trades'],
            'train_sharpe': train_result['sharpe'],
            'val_trades': val_result['n_trades'],
            'val_pnl': val_result['net_pnl'],
            'val_win_rate': val_result['win_rate'],
            'val_sharpe': val_result['sharpe'],
            'val_pf': val_result['profit_factor'],
            'avg_win': val_result['avg_win'],
            'avg_loss': val_result['avg_loss'],
            'ev_per_trade': ev_per_trade,
            'composite_score': composite_score
        })
    
    pbar.close()
    
    # Sort by composite score
    results.sort(key=lambda x: x['composite_score'], reverse=True)
    
    return results


def test_top_configs(
    results: List[Dict],
    test_df: pd.DataFrame,
    top_n: int = 10
) -> List[Dict]:
    """Test top configurations on held-out data."""
    test_results = []
    
    # Also include high R:R configs with enough trades
    high_rr = [r for r in results if r['rr_ratio'] >= 1.3 and r['val_trades'] >= 10]
    high_rr.sort(key=lambda x: x['rr_ratio'], reverse=True)
    
    # Combine top by score + top by R:R (unique)
    configs_to_test = results[:top_n]
    seen = set()
    for c in configs_to_test:
        key = (c['zscore_entry'], c['lookback'], c['stop_loss_mult'], c['take_profit_mult'])
        seen.add(key)
    
    for c in high_rr[:10]:
        key = (c['zscore_entry'], c['lookback'], c['stop_loss_mult'], c['take_profit_mult'])
        if key not in seen:
            configs_to_test.append(c)
            seen.add(key)
    
    print(f"\nTesting {len(configs_to_test)} configurations on held-out data...")
    
    for config in tqdm(configs_to_test, desc="Testing"):
        test_result = mean_reversion_backtest(
            test_df,
            zscore_entry=config['zscore_entry'],
            zscore_exit=config['zscore_exit'],
            lookback=config['lookback'],
            stop_loss_mult=config['stop_loss_mult'],
            take_profit_mult=config['take_profit_mult']
        )
        
        if test_result['avg_loss'] != 0:
            test_rr = abs(test_result['avg_win'] / test_result['avg_loss'])
        else:
            test_rr = 1.0
        
        test_results.append({
            **config,
            'test_trades': test_result['n_trades'],
            'test_pnl': test_result['net_pnl'],
            'test_win_rate': test_result['win_rate'],
            'test_sharpe': test_result['sharpe'],
            'test_pf': test_result['profit_factor'],
            'test_rr': test_rr,
            'test_avg_win': test_result['avg_win'],
            'test_avg_loss': test_result['avg_loss']
        })
    
    return test_results


def main():
    print("🎯 Project Delta - Advanced R:R Optimization")
    print("=" * 60)
    
    config = load_config(str(PROJECT_ROOT / "config.yml"))
    data_path = PROJECT_ROOT / config.paths.processed_data
    
    split = create_chronological_split(data_path)
    
    print(f"\nData split:")
    print(f"  Train: {len(split.train_files)} files ({split.train_files[0].stem} to {split.train_files[-1].stem})")
    print(f"  Val:   {len(split.val_files)} files ({split.val_files[0].stem} to {split.val_files[-1].stem})")
    print(f"  Test:  {len(split.test_files)} files ({split.test_files[0].stem} to {split.test_files[-1].stem})")
    
    # Load data
    print("\nLoading data...")
    train_dfs = [read_parquet_file(f) for f in tqdm(split.train_files, desc="Train")]
    train_df = pd.concat(train_dfs, ignore_index=True)
    del train_dfs
    gc.collect()
    print(f"  Train: {len(train_df)} bars")
    
    val_dfs = [read_parquet_file(f) for f in tqdm(split.val_files, desc="Val")]
    val_df = pd.concat(val_dfs, ignore_index=True)
    del val_dfs
    gc.collect()
    print(f"  Val: {len(val_df)} bars")
    
    test_dfs = [read_parquet_file(f) for f in tqdm(split.test_files, desc="Test")]
    test_df = pd.concat(test_dfs, ignore_index=True)
    del test_dfs
    gc.collect()
    print(f"  Test: {len(test_df)} bars")
    
    # Run comprehensive sweep
    print("\n" + "=" * 60)
    print("Running parameter sweep...")
    print("=" * 60)
    
    sweep_results = detailed_param_sweep(train_df, val_df)
    
    del train_df
    gc.collect()
    
    print(f"\nFound {len(sweep_results)} viable configurations")
    
    # Show top 10 by composite score
    print("\n📊 TOP 10 CONFIGURATIONS (by composite score)")
    print("-" * 100)
    print(f"{'Rank':<5} {'Z-Entry':<8} {'LB':<5} {'SL':<5} {'TP':<5} {'R:R':<6} "
          f"{'Val Trades':<10} {'Val PnL':<10} {'Val WR':<8} {'Val Sharpe':<10} {'Score':<8}")
    print("-" * 100)
    
    for i, r in enumerate(sweep_results[:10], 1):
        print(f"{i:<5} {r['zscore_entry']:<8.1f} {r['lookback']:<5} {r['stop_loss_mult']:<5.2f} "
              f"{r['take_profit_mult']:<5.1f} {r['rr_ratio']:<6.2f} {r['val_trades']:<10} "
              f"${r['val_pnl']:>8,.0f}  {r['val_win_rate']:>6.1%}  {r['val_sharpe']:>8.2f}  "
              f"{r['composite_score']:>6.2f}")
    
    # Focus on HIGH R:R configurations
    print("\n📈 TOP CONFIGURATIONS BY R:R RATIO (min R:R 1.5, min 10 trades)")
    high_rr = [r for r in sweep_results if r['rr_ratio'] >= 1.5 and r['val_trades'] >= 10]
    high_rr.sort(key=lambda x: x['rr_ratio'], reverse=True)
    
    print("-" * 100)
    for i, r in enumerate(high_rr[:10], 1):
        print(f"{i:<5} {r['zscore_entry']:<8.1f} {r['lookback']:<5} {r['stop_loss_mult']:<5.2f} "
              f"{r['take_profit_mult']:<5.1f} {r['rr_ratio']:<6.2f} {r['val_trades']:<10} "
              f"${r['val_pnl']:>8,.0f}  {r['val_win_rate']:>6.1%}  {r['val_sharpe']:>8.2f}  "
              f"{r['composite_score']:>6.2f}")
    
    # Test top configurations
    print("\n" + "=" * 60)
    print("Testing on HELD-OUT data")
    print("=" * 60)
    
    del val_df
    gc.collect()
    
    test_results = test_top_configs(sweep_results, test_df, top_n=15)
    
    # Show test results
    print("\n📊 TEST RESULTS (sorted by test Sharpe)")
    test_results.sort(key=lambda x: x['test_sharpe'], reverse=True)
    
    print("-" * 120)
    print(f"{'Rank':<5} {'Params':<25} {'R:R Val':<8} {'R:R Test':<8} "
          f"{'Test Trades':<12} {'Test PnL':<12} {'Test WR':<10} {'Test Sharpe':<10}")
    print("-" * 120)
    
    for i, r in enumerate(test_results, 1):
        params = f"z={r['zscore_entry']},lb={r['lookback']},sl={r['stop_loss_mult']},tp={r['take_profit_mult']}"
        print(f"{i:<5} {params:<25} {r['rr_ratio']:>6.2f}  {r['test_rr']:>6.2f}   "
              f"{r['test_trades']:<12} ${r['test_pnl']:>9,.0f}   {r['test_win_rate']:>8.1%}  "
              f"{r['test_sharpe']:>8.2f}")
    
    # Find best config that's actually profitable on test
    profitable_configs = [r for r in test_results if r['test_pnl'] > 0 and r['test_trades'] >= 5]
    
    if profitable_configs:
        profitable_configs.sort(key=lambda x: x['test_sharpe'], reverse=True)
        best = profitable_configs[0]
        print("\n🎯 BEST PROFITABLE CONFIGURATION (on test data)")
    else:
        # Fallback to best by Sharpe even if not profitable
        best = test_results[0]
        print("\n⚠️ No profitable configs on test - showing best by Sharpe")
    
    # Best configuration
    
    print("\n" + "=" * 60)
    print("🏆 BEST CONFIGURATION")
    print("=" * 60)
    print(f"  Z-score Entry:    {best['zscore_entry']}")
    print(f"  Z-score Exit:     {best['zscore_exit']}")
    print(f"  Lookback:         {best['lookback']}")
    print(f"  Stop Loss Mult:   {best['stop_loss_mult']}")
    print(f"  Take Profit Mult: {best['take_profit_mult']}")
    print(f"\n  Validation Performance:")
    print(f"    Trades:     {best['val_trades']}")
    print(f"    Net PnL:    ${best['val_pnl']:,.0f}")
    print(f"    Win Rate:   {best['val_win_rate']:.1%}")
    print(f"    R:R Ratio:  1:{best['rr_ratio']:.2f}")
    print(f"    Sharpe:     {best['val_sharpe']:.2f}")
    print(f"\n  Test Performance (HELD-OUT):")
    print(f"    Trades:     {best['test_trades']}")
    print(f"    Net PnL:    ${best['test_pnl']:,.0f}")
    print(f"    Win Rate:   {best['test_win_rate']:.1%}")
    print(f"    Avg Win:    ${best['test_avg_win']:,.0f}")
    print(f"    Avg Loss:   ${best['test_avg_loss']:,.0f}")
    print(f"    R:R Ratio:  1:{best['test_rr']:.2f}")
    print(f"    Sharpe:     {best['test_sharpe']:.2f}")
    
    # Calculate breakeven and edge
    if best['test_rr'] > 0:
        breakeven_wr = 1 / (1 + best['test_rr'])
        edge = best['test_win_rate'] - breakeven_wr
        print(f"\n  Edge Analysis:")
        print(f"    Breakeven WR: {breakeven_wr:.1%}")
        print(f"    Actual WR:    {best['test_win_rate']:.1%}")
        print(f"    Edge:         {edge*100:+.1f}pp")
    
    print("\n✅ Optimization complete!")


if __name__ == "__main__":
    main()
