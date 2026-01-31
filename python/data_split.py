"""
Data Splitting Strategy for Project Delta

Implements proper chronological train/validation/test splits
to prevent lookahead bias and overfitting.
"""

import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json


def read_parquet_file(path: Path) -> pd.DataFrame:
    """Read parquet file using Polars (for compatibility) and convert to pandas."""
    return pl.read_parquet(path).to_pandas()


@dataclass
class DataSplit:
    """Container for train/val/test file assignments."""
    train_files: List[Path]
    val_files: List[Path]
    test_files: List[Path]
    
    def __repr__(self):
        return (f"DataSplit(train={len(self.train_files)}, "
                f"val={len(self.val_files)}, test={len(self.test_files)})")


def create_chronological_split(
    data_dir: Path,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    min_train_files: int = 5,
    min_test_files: int = 2
) -> DataSplit:
    """
    Split data files chronologically.
    
    CRITICAL: We split by FILE (date) not by row to prevent any
    information leakage between train/val/test sets.
    
    Args:
        data_dir: Directory containing parquet files
        train_ratio: Fraction for training (earliest data)
        val_ratio: Fraction for validation (middle data)
        test_ratio: Fraction for testing (latest data)
        min_train_files: Minimum training files required
        min_test_files: Minimum test files required
    
    Returns:
        DataSplit object with file assignments
    """
    # Get all parquet files sorted by name (which should be date YYYYMMDD)
    files = sorted(data_dir.glob("*.parquet"))
    
    if not files:
        raise ValueError(f"No parquet files found in {data_dir}")
    
    n_files = len(files)
    
    if n_files < min_train_files + min_test_files:
        raise ValueError(f"Need at least {min_train_files + min_test_files} files, "
                        f"got {n_files}")
    
    # Calculate split indices
    n_train = max(min_train_files, int(n_files * train_ratio))
    n_test = max(min_test_files, int(n_files * test_ratio))
    n_val = n_files - n_train - n_test
    
    # Ensure we have validation data
    if n_val < 1:
        n_val = 1
        n_train = n_files - n_val - n_test
    
    # Split files chronologically
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    
    print(f"Data split created:")
    print(f"  Train: {len(train_files)} files ({files[0].stem} to {train_files[-1].stem})")
    print(f"  Val:   {len(val_files)} files ({val_files[0].stem} to {val_files[-1].stem})")
    print(f"  Test:  {len(test_files)} files ({test_files[0].stem} to {test_files[-1].stem})")
    
    return DataSplit(
        train_files=train_files,
        val_files=val_files,
        test_files=test_files
    )


def create_walk_forward_splits(
    data_dir: Path,
    n_splits: int = 5,
    train_window: int = 10,  # files
    test_window: int = 2,    # files
    expanding: bool = True   # expanding window vs rolling
) -> List[DataSplit]:
    """
    Create walk-forward validation splits.
    
    Walk-forward is the GOLD STANDARD for time series validation:
    - Train on historical data
    - Test on future data
    - Roll forward and repeat
    
    Args:
        data_dir: Directory containing parquet files
        n_splits: Number of train/test splits
        train_window: Number of files in training window
        test_window: Number of files in test window
        expanding: If True, training window grows; if False, it's fixed
    
    Returns:
        List of DataSplit objects for each walk-forward fold
    """
    files = sorted(data_dir.glob("*.parquet"))
    n_files = len(files)
    
    if n_files < train_window + test_window:
        raise ValueError(f"Need at least {train_window + test_window} files")
    
    splits = []
    
    # Calculate step size
    available_for_testing = n_files - train_window
    step_size = max(1, available_for_testing // n_splits)
    
    for i in range(n_splits):
        train_start = 0  # Default for expanding window
        if expanding:
            # Expanding window: train on all data up to split point
            train_end = train_window + i * step_size
        else:
            # Rolling window: fixed training window size
            train_start = i * step_size
            train_end = train_start + train_window
        
        test_start = train_end
        test_end = min(test_start + test_window, n_files)
        
        if test_start >= n_files:
            break
        
        if expanding:
            train_files = files[:train_end]
        else:
            train_files = files[train_start:train_end]
        
        test_files = files[test_start:test_end]
        
        # Use last 20% of train as validation (but still chronological)
        val_size = max(1, len(train_files) // 5)
        val_files = train_files[-val_size:]
        train_files = train_files[:-val_size]
        
        splits.append(DataSplit(
            train_files=list(train_files),
            val_files=list(val_files),
            test_files=list(test_files)
        ))
    
    print(f"Created {len(splits)} walk-forward splits:")
    for i, split in enumerate(splits):
        print(f"  Fold {i+1}: Train={len(split.train_files)}, "
              f"Val={len(split.val_files)}, Test={len(split.test_files)}")
    
    return splits


def load_split_data(
    split: DataSplit,
    set_type: str = 'train'
) -> pd.DataFrame:
    """
    Load data for a specific split set.
    
    Args:
        split: DataSplit object
        set_type: 'train', 'val', or 'test'
    
    Returns:
        Combined DataFrame
    """
    if set_type == 'train':
        files = split.train_files
    elif set_type == 'val':
        files = split.val_files
    elif set_type == 'test':
        files = split.test_files
    else:
        raise ValueError(f"Unknown set_type: {set_type}")
    
    dfs = []
    for f in files:
        df = read_parquet_file(f)
        df['source_file'] = f.stem
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


def save_split_config(split: DataSplit, output_path: Path):
    """Save split configuration for reproducibility."""
    config = {
        'train_files': [str(f) for f in split.train_files],
        'val_files': [str(f) for f in split.val_files],
        'test_files': [str(f) for f in split.test_files],
        'created_at': datetime.now().isoformat()
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_split_config(config_path: Path) -> DataSplit:
    """Load a saved split configuration."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return DataSplit(
        train_files=[Path(f) for f in config['train_files']],
        val_files=[Path(f) for f in config['val_files']],
        test_files=[Path(f) for f in config['test_files']]
    )
