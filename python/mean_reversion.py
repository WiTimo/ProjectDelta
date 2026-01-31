"""
Project Delta - Mean Reversion Strategy

Instead of predicting direction, trade when prices deviate from fair value.
Mean reversion tends to be more robust than directional prediction.
"""

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Tuple, List
from tqdm import tqdm
import gc

# Ensure we're using project root for relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

from config import load_config
from data_split import create_chronological_split


def get_project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def read_parquet_file(path: Path) -> pd.DataFrame:
    return pl.read_parquet(path).to_pandas()


def mean_reversion_backtest(
    df: pd.DataFrame,
    zscore_entry: float = 2.0,
    zscore_exit: float = 0.5,
    lookback: int = 50,
    max_hold_bars: int = 100,
    stop_loss_mult: float = 1.0,  # Stop loss as multiple of entry distance from MA
    take_profit_mult: float = 2.0,  # Take profit as multiple of entry distance from MA
    point_value: float = 20.0,
    cost_per_trade: float = 5.0
) -> dict:
    """
    Mean reversion strategy with PROPER RISK MANAGEMENT:
    - Fixed stop-loss to limit losses
    - Take profit target for asymmetric R:R
    - Exit on mean reversion OR hit targets
    """
    df = df.copy()
    
    # Filter out invalid prices (zeros, negatives)
    invalid_count = (df['close'] <= 0).sum()
    if invalid_count > 0:
        df = df[df['close'] > 0].reset_index(drop=True)
    
    # Calculate Z-score and MA
    ma = df['close'].rolling(lookback).mean()
    std = df['close'].rolling(lookback).std()
    df['zscore'] = (df['close'] - ma) / (std + 1e-8)
    df['ma'] = ma
    df['std'] = std
    
    # Drop NaN from rolling
    df = df.dropna().reset_index(drop=True)
    
    # Convert to numpy arrays for speed (avoid df.iloc overhead)
    prices = df['close'].values
    zscores = df['zscore'].values
    mas = df['ma'].values
    n = len(prices)
    
    trades = []
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_bar = 0
    stop_price = 0.0
    target_price = 0.0
    
    for i in range(n):
        z = zscores[i]
        price = prices[i]
        current_ma = mas[i]
        
        if position == 0:
            # Look for entry
            if z > zscore_entry:
                # Price too high, go short (expecting reversion to mean)
                position = -1
                entry_price = price
                entry_bar = i
                
                # Distance from mean (this is our expected profit)
                distance_to_mean = price - current_ma
                
                # Stop loss: price goes FURTHER from mean (we're wrong)
                stop_price = price + (distance_to_mean * stop_loss_mult)
                
                # Take profit: price overshoots mean (we're right)
                target_price = current_ma - (distance_to_mean * (take_profit_mult - 1))
                
            elif z < -zscore_entry:
                # Price too low, go long (expecting reversion to mean)
                position = 1
                entry_price = price
                entry_bar = i
                
                # Distance from mean
                distance_to_mean = current_ma - price
                
                # Stop loss: price goes further down (we're wrong)
                stop_price = price - (distance_to_mean * stop_loss_mult)
                
                # Take profit: price overshoots mean (we're right)
                target_price = current_ma + (distance_to_mean * (take_profit_mult - 1))
                
        else:
            # Check exit conditions
            bars_held = i - entry_bar
            exit_signal = False
            exit_reason = None
            
            if position == 1:  # Long position
                if price <= stop_price:
                    exit_signal = True
                    exit_reason = 'stop_loss'
                elif price >= target_price:
                    exit_signal = True
                    exit_reason = 'take_profit'
                elif z >= -zscore_exit:  # Reverted to mean
                    exit_signal = True
                    exit_reason = 'mean_reversion'
                    
            elif position == -1:  # Short position
                if price >= stop_price:
                    exit_signal = True
                    exit_reason = 'stop_loss'
                elif price <= target_price:
                    exit_signal = True
                    exit_reason = 'take_profit'
                elif z <= zscore_exit:  # Reverted to mean
                    exit_signal = True
                    exit_reason = 'mean_reversion'
            
            # Force exit on max hold
            if bars_held >= max_hold_bars:
                exit_signal = True
                exit_reason = 'max_hold'
            
            if exit_signal:
                exit_price = price
                pnl_points = (exit_price - entry_price) * position
                pnl_dollars = pnl_points * point_value - cost_per_trade
                
                trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'side': 'long' if position == 1 else 'short',
                    'pnl': pnl_dollars,
                    'bars_held': bars_held,
                    'exit_reason': exit_reason
                })
                
                position = 0
    
    # Calculate metrics
    if not trades:
        return {'n_trades': 0, 'pnl': 0, 'sharpe': 0, 'win_rate': 0}
    
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    
    net_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls)
    avg_trade = np.mean(pnls)
    
    # Sharpe (simplified)
    if np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252)  # Annualized
    else:
        sharpe = 0
    
    # Profit factor
    if losses and sum(losses) != 0:
        pf = sum(wins) / abs(sum(losses)) if wins else 0
    else:
        pf = float('inf') if wins else 0
    
    return {
        'n_trades': len(trades),
        'net_pnl': net_pnl,
        'win_rate': win_rate,
        'avg_trade': avg_trade,
        'avg_win': np.mean(wins) if wins else 0,
        'avg_loss': np.mean(losses) if losses else 0,
        'sharpe': sharpe,
        'profit_factor': pf,
        'trades': trades
    }


def main():
    print("📉 Project Delta - Mean Reversion Strategy")
    print("=" * 60)
    
    config = load_config(str(PROJECT_ROOT / "config.yml"))
    data_path = get_project_path(config.paths.processed_data)
    
    # Get data split
    labeled_dir = data_path / "labeled"
    if labeled_dir.exists() and len(list(labeled_dir.glob("*.parquet"))) > 10:
        pass
    else:
        labeled_dir = data_path
    
    split = create_chronological_split(labeled_dir)
    
    print(f"\nData split:")
    print(f"  Train: {len(split.train_files)} files")
    print(f"  Val:   {len(split.val_files)} files")
    print(f"  Test:  {len(split.test_files)} files")
    
    # Load validation data for parameter optimization
    print("\nLoading validation data...")
    val_dfs = []
    for f in tqdm(split.val_files, desc="Val"):
        val_dfs.append(read_parquet_file(f))
    val_df = pd.concat(val_dfs, ignore_index=True)
    del val_dfs
    gc.collect()
    print(f"  Loaded {len(val_df)} bars")
    
    # Parameter sweep on validation - NOW INCLUDING RISK:REWARD
    print("\nOptimizing parameters on validation data...")
    print("  Including risk:reward ratio optimization...")
    best_params = None
    best_sharpe = -np.inf
    
    for zscore_entry in [2.0, 2.5, 3.0]:
        for lookback in [30, 50]:
            for stop_loss_mult in [0.5, 0.75, 1.0]:  # Tighter stops
                for take_profit_mult in [1.5, 2.0, 2.5]:  # Larger targets
                    # Skip if R:R ratio is less than 1.0
                    if take_profit_mult <= stop_loss_mult:
                        continue
                        
                    result = mean_reversion_backtest(
                        val_df,
                        zscore_entry=zscore_entry,
                        zscore_exit=0.25,  # Fixed
                        lookback=lookback,
                        stop_loss_mult=stop_loss_mult,
                        take_profit_mult=take_profit_mult
                    )
                    
                    if result['n_trades'] >= 15 and result['sharpe'] > best_sharpe:
                        best_sharpe = result['sharpe']
                        best_params = {
                            'zscore_entry': zscore_entry,
                            'zscore_exit': 0.25,
                            'lookback': lookback,
                            'stop_loss_mult': stop_loss_mult,
                            'take_profit_mult': take_profit_mult
                        }
                        best_result = result
    
    if best_params:
        print(f"\nBest parameters (Validation):")
        print(f"  Z-score entry:    {best_params['zscore_entry']}")
        print(f"  Lookback:         {best_params['lookback']}")
        print(f"  Stop loss mult:   {best_params['stop_loss_mult']}")
        print(f"  Take profit mult: {best_params['take_profit_mult']}")
        print(f"  Risk:Reward:      1:{best_params['take_profit_mult']/best_params['stop_loss_mult']:.1f}")
        print(f"\nValidation Results:")
        print(f"  Trades:        {best_result['n_trades']}")
        print(f"  Net PnL:       ${best_result['net_pnl']:,.2f}")
        print(f"  Win Rate:      {best_result['win_rate']:.1%}")
        print(f"  Avg Win:       ${best_result['avg_win']:,.2f}")
        print(f"  Avg Loss:      ${best_result['avg_loss']:,.2f}")
        print(f"  Sharpe:        {best_result['sharpe']:.2f}")
        print(f"  Profit Factor: {best_result['profit_factor']:.2f}")
        
        # Show exit reason breakdown
        if 'trades' in best_result:
            reasons = {}
            for t in best_result['trades']:
                r = t.get('exit_reason', 'unknown')
                reasons[r] = reasons.get(r, 0) + 1
            print(f"\n  Exit Reasons:")
            for r, count in sorted(reasons.items()):
                print(f"    {r}: {count}")
    else:
        print("No viable parameters found on validation data")
        # Use default with good R:R
        best_params = {
            'zscore_entry': 2.0, 
            'zscore_exit': 0.25, 
            'lookback': 50,
            'stop_loss_mult': 0.75,
            'take_profit_mult': 1.5
        }
    
    # Test on held-out data
    print("\n" + "=" * 60)
    print("Testing on HELD-OUT data")
    print("=" * 60)
    
    test_dfs = []
    for f in tqdm(split.test_files, desc="Test"):
        test_dfs.append(read_parquet_file(f))
    test_df = pd.concat(test_dfs, ignore_index=True)
    del test_dfs
    gc.collect()
    print(f"  Loaded {len(test_df)} bars")
    
    test_result = mean_reversion_backtest(
        test_df,
        **best_params
    )
    
    print(f"\n📊 HELD-OUT TEST RESULTS")
    print(f"  Trades:        {test_result['n_trades']}")
    print(f"  Net PnL:       ${test_result['net_pnl']:,.2f}")
    print(f"  Win Rate:      {test_result['win_rate']:.1%}")
    print(f"  Avg Trade:     ${test_result['avg_trade']:,.2f}")
    print(f"  Avg Win:       ${test_result['avg_win']:,.2f}")
    print(f"  Avg Loss:      ${test_result['avg_loss']:,.2f}")
    
    # Risk:Reward analysis
    if test_result['avg_loss'] != 0:
        rr_ratio = abs(test_result['avg_win'] / test_result['avg_loss'])
        print(f"  Risk:Reward:   1:{rr_ratio:.2f}")
        
        # Calculate what win rate is needed for breakeven with this R:R
        breakeven_wr = 1 / (1 + rr_ratio)
        print(f"  Breakeven WR:  {breakeven_wr:.1%} (need to beat this)")
        
        if test_result['win_rate'] > breakeven_wr:
            print(f"  ✓ Win rate beats breakeven by {(test_result['win_rate'] - breakeven_wr)*100:.1f}pp")
        else:
            print(f"  ✗ Win rate below breakeven by {(breakeven_wr - test_result['win_rate'])*100:.1f}pp")
    
    print(f"  Sharpe:        {test_result['sharpe']:.2f}")
    print(f"  Profit Factor: {test_result['profit_factor']:.2f}")
    
    # Exit reason breakdown
    if 'trades' in test_result and test_result['trades']:
        reasons = {}
        for t in test_result['trades']:
            r = t.get('exit_reason', 'unknown')
            reasons[r] = reasons.get(r, 0) + 1
        print(f"\n  Exit Reasons:")
        for r, count in sorted(reasons.items()):
            pct = count / len(test_result['trades']) * 100
            print(f"    {r}: {count} ({pct:.0f}%)")
    
    # Quality check
    print("\n✅ Quality Assessment:")
    if test_result['sharpe'] >= 0.5:
        print("  ✓ Sharpe >= 0.5 (decent risk-adjusted returns)")
    else:
        print("  ✗ Sharpe < 0.5 (poor risk-adjusted returns)")
    
    if test_result['profit_factor'] >= 1.0:
        print("  ✓ Profit factor >= 1.0 (profitable)")
    else:
        print("  ✗ Profit factor < 1.0 (losing)")
    
    if test_result['win_rate'] >= 0.4:
        print("  ✓ Win rate >= 40%")
    else:
        print("  ✗ Win rate < 40%")
    
    # Compare val vs test
    print("\n📈 Validation vs Test Comparison:")
    if best_params and 'sharpe' in best_result:
        sharpe_diff = test_result['sharpe'] - best_result['sharpe']
        print(f"  Sharpe change: {sharpe_diff:+.2f}")
        if abs(sharpe_diff) < 0.5:
            print("  ✓ Reasonable stability between val and test")
        else:
            print("  ⚠ Large gap suggests overfitting to validation period")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
