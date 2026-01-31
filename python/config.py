"""
Configuration loader for Project Delta - Mean Reversion Strategy
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field


from typing import Optional


@dataclass
class PathsConfig:
    """Data paths configuration."""
    raw_data: str = "data/raw"
    processed_data: str = "data/processed"


@dataclass
class PreprocessingConfig:
    """Rust preprocessor configuration."""
    volume_bar_size: int = 500
    max_levels: int = 10
    ofi_levels: int = 5


@dataclass
class TradingConfig:
    """Trading cost and contract configuration."""
    point_value: float = 20.0
    tick_size: float = 0.25
    commission_per_trade: float = 5.0
    slippage_ticks: int = 1
    spread_ticks: int = 1


@dataclass
class StrategyConfig:
    """Mean reversion strategy parameters."""
    zscore_entry: float = 3.0
    zscore_exit: float = 0.5
    lookback: int = 50
    stop_loss_mult: float = 0.5
    take_profit_mult: float = 1.5
    max_hold_bars: int = 100
    contracts: int = 1
    max_daily_trades: int = 20
    max_daily_loss: float = 500.0


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    execution_delay_bars: int = 1
    use_slippage: bool = True
    use_spread: bool = True
    train_window_days: int = 15
    test_window_days: int = 5
    step_size_days: int = 5


@dataclass
class Config:
    """Main configuration container."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, searches for config.yml
                    in current directory and parent directories.
    
    Returns:
        Config object with loaded settings.
    """
    if config_path is None:
        # Search for config.yml in current and parent directories
        search_paths = [
            Path.cwd() / "config.yml",
            Path.cwd().parent / "config.yml",
            Path(__file__).parent.parent / "config.yml",
        ]
        for path in search_paths:
            if path.exists():
                config_path = str(path)
                break
    
    path = Path(config_path) if config_path else None
    
    if path is None or not path.exists():
        print("Warning: Config file not found, using defaults")
        return Config()
    
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    
    config = Config()
    
    # Parse each section if present
    if 'paths' in raw:
        config.paths = PathsConfig(**raw['paths'])
    
    if 'preprocessing' in raw:
        # Handle nested trading_hours by excluding it
        prep = {k: v for k, v in raw['preprocessing'].items() 
                if k not in ['trading_hours', 'volume_bar']}
        if 'volume_bar_size' in prep:
            config.preprocessing = PreprocessingConfig(**prep)
    
    if 'trading' in raw:
        config.trading = TradingConfig(**raw['trading'])
    
    if 'strategy' in raw:
        config.strategy = StrategyConfig(**raw['strategy'])
    
    if 'backtesting' in raw:
        config.backtest = BacktestConfig(**raw['backtesting'])
    
    return config


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


if __name__ == "__main__":
    config = load_config()
    print(f"Paths: {config.paths}")
    print(f"Preprocessing: {config.preprocessing}")
    print(f"Trading: {config.trading}")
    print(f"Strategy: {config.strategy}")
    print(f"Backtest: {config.backtest}")
