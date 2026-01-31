//! Configuration handling for the preprocessor

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

// Allow unused fields - they're read from YAML config for future use
#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub data: DataConfig,
    pub preprocessing: PreprocessingConfig,
    pub labeling: LabelingConfig,
    pub features: FeaturesConfig,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
pub struct DataConfig {
    pub raw_dir: String,
    pub preprocessed_dir: String,
    pub models_dir: String,
    pub results_dir: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
pub struct PreprocessingConfig {
    pub volume_bar: VolumeBarConfig,
    pub order_book: OrderBookConfig,
    pub trading_hours: TradingHoursConfig,
    pub output: OutputConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct VolumeBarConfig {
    pub target_volume: u64,
    pub strict_split: bool,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OrderBookConfig {
    pub max_levels: usize,
    pub ofi_levels: usize,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
pub struct TradingHoursConfig {
    pub enabled: bool,
    pub start_time: String,
    pub end_time: String,
    pub include_premarket: bool,
    pub premarket_start: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
pub struct OutputConfig {
    pub format: String,
    pub compression: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
pub struct LabelingConfig {
    pub triple_barrier: TripleBarrierConfig,
    pub classes: ClassesConfig,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
pub struct TripleBarrierConfig {
    pub volatility_window: usize,
    pub upper_barrier_mult: f64,
    pub lower_barrier_mult: f64,
    pub vertical_barrier_bars: usize,
    pub min_volatility: f64,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
pub struct ClassesConfig {
    pub balance_method: String,
    pub max_neutral_ratio: f64,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
pub struct FeaturesConfig {
    pub lag_periods: Vec<usize>,
    pub rolling_windows: Vec<usize>,
    pub include: Vec<String>,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {:?}", path))?;
        
        let config: Config = serde_yaml::from_str(&content)
            .with_context(|| "Failed to parse config YAML")?;
        
        Ok(config)
    }
}
