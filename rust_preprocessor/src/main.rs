//! Project Delta - NQ Microstructure Preprocessor
//! 
//! High-performance Rust preprocessor for converting raw NinjaTrader L1/L2 tick data
//! into Volume Bars with microstructure features (OFI, Trade Imbalance, etc.)

mod config;
mod parser;
mod order_book;
mod volume_bar;
mod features;
mod output;

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use tracing::{info, warn, error, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to config.yml
    #[arg(short, long, default_value = "config.yml")]
    config: PathBuf,

    /// Override raw data directory
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Override output directory
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Process single file only
    #[arg(short, long)]
    file: Option<PathBuf>,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.verbose { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .with_target(false)
        .with_thread_ids(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("🚀 Project Delta - NQ Microstructure Preprocessor");
    info!("Loading configuration from: {:?}", args.config);

    // Load configuration
    let config = config::Config::load(&args.config)
        .context("Failed to load configuration")?;

    // Determine input/output directories
    let raw_dir = args.input.unwrap_or_else(|| PathBuf::from(&config.data.raw_dir));
    let output_dir = args.output.unwrap_or_else(|| PathBuf::from(&config.data.preprocessed_dir));

    // Create output directory
    std::fs::create_dir_all(&output_dir)
        .context("Failed to create output directory")?;

    // Get list of files to process
    let files: Vec<PathBuf> = if let Some(single_file) = args.file {
        vec![single_file]
    } else {
        std::fs::read_dir(&raw_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "csv"))
            .collect()
    };

    if files.is_empty() {
        warn!("No CSV files found in {:?}", raw_dir);
        return Ok(());
    }

    info!("Found {} files to process", files.len());

    // Process files SEQUENTIALLY to minimize RAM usage
    // Each file is fully processed and written before loading the next
    let mut total_bars = 0u64;
    let mut total_ticks = 0u64;
    let mut errors = 0u64;

    for (idx, file) in files.iter().enumerate() {
        info!("Processing file {}/{}: {:?}", idx + 1, files.len(), file.file_name().unwrap_or_default());
        
        let result = process_file(file, &output_dir, &config);
        match result {
            Ok(stats) => {
                total_bars += stats.bars_created;
                total_ticks += stats.ticks_processed;
            }
            Err(e) => {
                error!("Processing error: {}", e);
                errors += 1;
            }
        }
    }

    info!("✅ Processing complete!");
    info!("   Total ticks processed: {}", total_ticks);
    info!("   Total volume bars created: {}", total_bars);
    if errors > 0 {
        warn!("   Files with errors: {}", errors);
    }

    Ok(())
}

#[derive(Debug, Default)]
struct ProcessingStats {
    ticks_processed: u64,
    bars_created: u64,
}

fn process_file(
    input_path: &PathBuf,
    output_dir: &PathBuf,
    config: &config::Config,
) -> Result<ProcessingStats> {
    let filename = input_path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    
    info!("Processing: {}", filename);

    // Initialize components
    let mut order_book = order_book::LimitOrderBook::new(config.preprocessing.order_book.max_levels);
    let mut bar_builder = volume_bar::VolumeBarBuilder::new(
        config.preprocessing.volume_bar.target_volume,
        config.preprocessing.volume_bar.strict_split,
    );
    let mut feature_calculator = features::FeatureCalculator::new(
        config.preprocessing.order_book.ofi_levels,
    );

    // Parse and process file
    let mut stats = ProcessingStats::default();
    let mut completed_bars = Vec::new();

    // Stream file line-by-line to minimize RAM usage (instead of loading entire file)
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(input_path)
        .context("Failed to open input file")?;
    let reader = BufReader::with_capacity(64 * 1024, file); // 64KB buffer

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => continue,
        };
        if line.is_empty() {
            continue;
        }

        match parser::parse_record(&line) {
            Ok(record) => {
                stats.ticks_processed += 1;

                // Update order book state
                let book_update = order_book.update(&record);

                // Accumulate features
                feature_calculator.accumulate(&record, &book_update, &order_book);

                // Check for trade (L1 Last)
                if let parser::Record::L1 { data_type: parser::MarketDataType::Last, volume, .. } = &record {
                    // Add volume to bar builder
                    let filled_bars = bar_builder.add_volume(
                        *volume,
                        &record,
                        &order_book,
                        &mut feature_calculator,
                    );

                    completed_bars.extend(filled_bars);
                }
            }
            Err(e) => {
                // Skip malformed lines silently (common in tick data)
                tracing::trace!("Parse error: {} - line: {}", e, line);
            }
        }
    }

    stats.bars_created = completed_bars.len() as u64;

    // Write output
    if !completed_bars.is_empty() {
        let output_path = output_dir.join(format!("{}.parquet", filename));
        output::write_parquet(&completed_bars, &output_path, config)?;
        info!("  Created {} volume bars -> {:?}", stats.bars_created, output_path);
    } else {
        warn!("  No volume bars created for {}", filename);
    }

    Ok(stats)
}
