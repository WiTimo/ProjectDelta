//! Output writers for volume bar data

use crate::config::Config;
use crate::features::BarFeatures;
use anyhow::{Context, Result};
use polars::prelude::*;
use std::path::Path;

/// Write volume bars to Parquet file
pub fn write_parquet(bars: &[BarFeatures], path: &Path, _config: &Config) -> Result<()> {
    if bars.is_empty() {
        return Ok(());
    }

    // Filter out invalid bars (zero prices, no trades, etc.)
    let valid_bars: Vec<&BarFeatures> = bars.iter()
        .filter(|b| {
            b.open > 0.0 && b.high > 0.0 && b.low > 0.0 && b.close > 0.0 &&
            b.trade_count > 0 &&
            b.high >= b.low && b.high >= b.open && b.high >= b.close &&
            b.low <= b.open && b.low <= b.close
        })
        .collect();
    
    let filtered_count = bars.len() - valid_bars.len();
    if filtered_count > 0 {
        tracing::warn!("Filtered out {} invalid bars (zero prices or bad OHLC)", filtered_count);
    }
    
    if valid_bars.is_empty() {
        return Ok(());
    }

    // Convert to columnar format
    let n = valid_bars.len();
    
    let timestamp_start: Vec<i64> = valid_bars.iter().map(|b| b.timestamp_start).collect();
    let timestamp_end: Vec<i64> = valid_bars.iter().map(|b| b.timestamp_end).collect();
    
    let open: Vec<f64> = valid_bars.iter().map(|b| b.open).collect();
    let high: Vec<f64> = valid_bars.iter().map(|b| b.high).collect();
    let low: Vec<f64> = valid_bars.iter().map(|b| b.low).collect();
    let close: Vec<f64> = valid_bars.iter().map(|b| b.close).collect();
    let volume: Vec<u64> = valid_bars.iter().map(|b| b.volume).collect();
    let trade_count: Vec<u32> = valid_bars.iter().map(|b| b.trade_count).collect();
    
    let ofi_level1: Vec<f64> = valid_bars.iter().map(|b| b.ofi_level1).collect();
    let ofi_aggregate: Vec<f64> = valid_bars.iter().map(|b| b.ofi_aggregate).collect();
    
    let buy_volume: Vec<u64> = valid_bars.iter().map(|b| b.buy_volume).collect();
    let sell_volume: Vec<u64> = valid_bars.iter().map(|b| b.sell_volume).collect();
    let trade_imbalance: Vec<f64> = valid_bars.iter().map(|b| b.trade_imbalance).collect();
    
    let depth_ratio: Vec<f64> = valid_bars.iter().map(|b| b.depth_ratio).collect();
    let spread_avg: Vec<f64> = valid_bars.iter().map(|b| b.spread_avg).collect();
    let spread_max: Vec<f64> = valid_bars.iter().map(|b| b.spread_max).collect();
    
    let sweep_count: Vec<u32> = valid_bars.iter().map(|b| b.sweep_count).collect();
    let aggressive_buy_count: Vec<u32> = valid_bars.iter().map(|b| b.aggressive_buy_count).collect();
    let aggressive_sell_count: Vec<u32> = valid_bars.iter().map(|b| b.aggressive_sell_count).collect();
    
    let vwap: Vec<f64> = valid_bars.iter().map(|b| b.vwap).collect();
    let vwap_deviation: Vec<f64> = valid_bars.iter().map(|b| b.vwap_deviation).collect();
    
    let price_delta: Vec<f64> = valid_bars.iter().map(|b| b.price_delta).collect();
    let high_low_range: Vec<f64> = valid_bars.iter().map(|b| b.high_low_range).collect();
    
    let bid_close: Vec<f64> = valid_bars.iter().map(|b| b.bid_close).collect();
    let ask_close: Vec<f64> = valid_bars.iter().map(|b| b.ask_close).collect();
    let mid_close: Vec<f64> = valid_bars.iter().map(|b| b.mid_close).collect();

    // Create bar index
    let bar_index: Vec<u64> = (0..n as u64).collect();

    // Build DataFrame
    let df = DataFrame::new(vec![
        Series::new("bar_index".into(), bar_index),
        Series::new("timestamp_start".into(), timestamp_start),
        Series::new("timestamp_end".into(), timestamp_end),
        Series::new("open".into(), open),
        Series::new("high".into(), high),
        Series::new("low".into(), low),
        Series::new("close".into(), close),
        Series::new("volume".into(), volume),
        Series::new("trade_count".into(), trade_count),
        Series::new("ofi_level1".into(), ofi_level1),
        Series::new("ofi_aggregate".into(), ofi_aggregate),
        Series::new("buy_volume".into(), buy_volume),
        Series::new("sell_volume".into(), sell_volume),
        Series::new("trade_imbalance".into(), trade_imbalance),
        Series::new("depth_ratio".into(), depth_ratio),
        Series::new("spread_avg".into(), spread_avg),
        Series::new("spread_max".into(), spread_max),
        Series::new("sweep_count".into(), sweep_count),
        Series::new("aggressive_buy_count".into(), aggressive_buy_count),
        Series::new("aggressive_sell_count".into(), aggressive_sell_count),
        Series::new("vwap".into(), vwap),
        Series::new("vwap_deviation".into(), vwap_deviation),
        Series::new("price_delta".into(), price_delta),
        Series::new("high_low_range".into(), high_low_range),
        Series::new("bid_close".into(), bid_close),
        Series::new("ask_close".into(), ask_close),
        Series::new("mid_close".into(), mid_close),
    ])
    .context("Failed to create DataFrame")?;

    // Write to Parquet
    let file = std::fs::File::create(path)
        .context("Failed to create output file")?;
    
    let mut buf_writer = std::io::BufWriter::new(file);
    
    ParquetWriter::new(&mut buf_writer)
        .with_compression(ParquetCompression::Snappy)
        .finish(&mut df.clone())
        .context("Failed to write Parquet file")?;
    
    // Ensure all data is flushed to disk
    use std::io::Write;
    buf_writer.flush().context("Failed to flush Parquet file")?;

    Ok(())
}

/// Write volume bars to CSV file (alternative format)
#[allow(dead_code)]
pub fn write_csv(bars: &[BarFeatures], path: &Path) -> Result<()> {
    use std::io::Write;
    
    let mut file = std::fs::File::create(path)?;
    
    // Header
    writeln!(file, "bar_index,timestamp_start,timestamp_end,open,high,low,close,volume,trade_count,ofi_level1,ofi_aggregate,buy_volume,sell_volume,trade_imbalance,depth_ratio,spread_avg,spread_max,sweep_count,aggressive_buy_count,aggressive_sell_count,vwap,vwap_deviation,price_delta,high_low_range,bid_close,ask_close,mid_close")?;
    
    for (i, bar) in bars.iter().enumerate() {
        writeln!(
            file,
            "{},{},{},{:.4},{:.4},{:.4},{:.4},{},{},{:.4},{:.4},{},{},{:.6},{:.4},{:.4},{:.4},{},{},{},{:.4},{:.6},{:.4},{:.4},{:.4},{:.4},{:.4}",
            i,
            bar.timestamp_start,
            bar.timestamp_end,
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
            bar.trade_count,
            bar.ofi_level1,
            bar.ofi_aggregate,
            bar.buy_volume,
            bar.sell_volume,
            bar.trade_imbalance,
            bar.depth_ratio,
            bar.spread_avg,
            bar.spread_max,
            bar.sweep_count,
            bar.aggressive_buy_count,
            bar.aggressive_sell_count,
            bar.vwap,
            bar.vwap_deviation,
            bar.price_delta,
            bar.high_low_range,
            bar.bid_close,
            bar.ask_close,
            bar.mid_close,
        )?;
    }
    
    Ok(())
}
