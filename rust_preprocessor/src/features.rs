//! Feature calculation for microstructure analysis

use crate::order_book::{BookUpdate, LimitOrderBook, Side};
use crate::parser::{MarketDataType, Operation, Record};

/// Accumulated features for a volume bar
#[derive(Debug, Clone, Default)]
pub struct BarFeatures {
    // Timing
    pub timestamp_start: i64,
    pub timestamp_end: i64,
    
    // OHLCV
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: u64,
    pub trade_count: u32,
    
    // Order Flow Imbalance
    pub ofi_level1: f64,
    pub ofi_aggregate: f64,
    
    // Trade Imbalance (VPIN-like)
    pub buy_volume: u64,
    pub sell_volume: u64,
    pub trade_imbalance: f64,
    
    // Order Book Shape
    pub depth_ratio: f64,
    pub spread_avg: f64,
    pub spread_max: f64,
    
    // Sweep Indicator
    pub sweep_count: u32,
    pub aggressive_buy_count: u32,
    pub aggressive_sell_count: u32,
    
    // VWAP
    pub vwap: f64,
    pub vwap_deviation: f64,
    
    // Price dynamics
    pub price_delta: f64,
    pub high_low_range: f64,
    
    // Bid/Ask at close
    pub bid_close: f64,
    pub ask_close: f64,
    pub mid_close: f64,
}

/// Feature calculator that accumulates features across ticks
#[derive(Debug)]
pub struct FeatureCalculator {
    ofi_levels: usize,
    
    // Accumulators
    ofi_level1_acc: f64,
    ofi_aggregate_acc: f64,
    
    buy_volume_acc: u64,
    sell_volume_acc: u64,
    
    spread_sum: f64,
    spread_count: u32,
    spread_max: f64,
    
    depth_ratio_sum: f64,
    depth_ratio_count: u32,
    
    sweep_count: u32,
    aggressive_buy_count: u32,
    aggressive_sell_count: u32,
    
    // VWAP
    price_volume_sum: f64,
    volume_sum: u64,
    
    // OHLC tracking
    open_price: f64,
    high_price: f64,
    low_price: f64,
    close_price: f64,
    trade_count: u32,
    
    // Timing
    first_timestamp: i64,
    last_timestamp: i64,
    
    // L2 removal tracking for sweep detection
    recent_l2_removes: Vec<(i64, Side, u64)>,
}

impl FeatureCalculator {
    pub fn new(ofi_levels: usize) -> Self {
        Self {
            ofi_levels,
            ofi_level1_acc: 0.0,
            ofi_aggregate_acc: 0.0,
            buy_volume_acc: 0,
            sell_volume_acc: 0,
            spread_sum: 0.0,
            spread_count: 0,
            spread_max: 0.0,
            depth_ratio_sum: 0.0,
            depth_ratio_count: 0,
            sweep_count: 0,
            aggressive_buy_count: 0,
            aggressive_sell_count: 0,
            price_volume_sum: 0.0,
            volume_sum: 0,
            open_price: 0.0,
            high_price: f64::MIN,
            low_price: f64::MAX,
            close_price: 0.0,
            trade_count: 0,
            first_timestamp: 0,
            last_timestamp: 0,
            recent_l2_removes: Vec::new(),
        }
    }

    /// Accumulate features from a single tick
    pub fn accumulate(&mut self, record: &Record, update: &BookUpdate, book: &LimitOrderBook) {
        let ts = record.timestamp_nanos();
        
        if self.first_timestamp == 0 {
            self.first_timestamp = ts;
        }
        self.last_timestamp = ts;

        // Process based on record type
        match record {
            Record::L1 { data_type, price, volume, .. } => {
                match data_type {
                    MarketDataType::Last => {
                        // Trade occurred
                        self.process_trade(*price, *volume, book, ts);
                    }
                    MarketDataType::Bid | MarketDataType::Ask => {
                        // Quote update - calculate OFI
                        self.calculate_ofi(update, book);
                        
                        // Track spread
                        let spread = book.spread();
                        if spread > 0.0 {
                            self.spread_sum += spread;
                            self.spread_count += 1;
                            if spread > self.spread_max {
                                self.spread_max = spread;
                            }
                        }
                        
                        // Track depth ratio
                        let ratio = book.depth_ratio(self.ofi_levels);
                        if ratio > 0.0 && ratio < 100.0 {
                            self.depth_ratio_sum += ratio;
                            self.depth_ratio_count += 1;
                        }
                    }
                    _ => {}
                }
            }
            Record::L2 { data_type, operation, volume, .. } => {
                // Track L2 removes for sweep detection
                if *operation == Operation::Remove && *volume > 0 {
                    let side = match data_type {
                        MarketDataType::Bid => Side::Bid,
                        MarketDataType::Ask => Side::Ask,
                        _ => return,
                    };
                    self.recent_l2_removes.push((ts, side, *volume));
                    
                    // Clean old removes (older than 100ms)
                    self.recent_l2_removes.retain(|(t, _, _)| ts - t < 100_000_000);
                }
            }
        }
    }

    /// Process a trade tick
    fn process_trade(&mut self, price: f64, volume: u64, book: &LimitOrderBook, ts: i64) {
        // OHLC
        if self.open_price == 0.0 {
            self.open_price = price;
        }
        if price > self.high_price {
            self.high_price = price;
        }
        if price < self.low_price {
            self.low_price = price;
        }
        self.close_price = price;
        self.trade_count += 1;

        // VWAP accumulation
        self.price_volume_sum += price * volume as f64;
        self.volume_sum += volume;

        // Trade classification (Lee-Ready style)
        // If trade at ask or above -> buy
        // If trade at bid or below -> sell
        // If between, use tick test
        if price >= book.best_ask && book.best_ask > 0.0 {
            self.buy_volume_acc += volume;
            self.aggressive_buy_count += 1;
            
            // Check for sweep (multiple L2 removes on ask side recently)
            let ask_removes: u64 = self.recent_l2_removes
                .iter()
                .filter(|(t, side, _)| ts - t < 50_000_000 && *side == Side::Ask)
                .map(|(_, _, v)| v)
                .sum();
            if ask_removes > 10 {
                self.sweep_count += 1;
            }
        } else if price <= book.best_bid && book.best_bid > 0.0 {
            self.sell_volume_acc += volume;
            self.aggressive_sell_count += 1;
            
            // Check for sweep on bid side
            let bid_removes: u64 = self.recent_l2_removes
                .iter()
                .filter(|(t, side, _)| ts - t < 50_000_000 && *side == Side::Bid)
                .map(|(_, _, v)| v)
                .sum();
            if bid_removes > 10 {
                self.sweep_count += 1;
            }
        } else {
            // Mid-spread trade - split volume
            self.buy_volume_acc += volume / 2;
            self.sell_volume_acc += volume - volume / 2;
        }
    }

    /// Calculate Order Flow Imbalance
    fn calculate_ofi(&mut self, update: &BookUpdate, book: &LimitOrderBook) {
        // Level 1 OFI based on best bid/ask changes
        let bid_ofi = self.calculate_side_ofi(
            book.prev_best_bid,
            book.prev_best_bid_size,
            book.best_bid,
            book.best_bid_size,
        );
        
        let ask_ofi = self.calculate_side_ofi(
            book.prev_best_ask,
            book.prev_best_ask_size,
            book.best_ask,
            book.best_ask_size,
        );
        
        // OFI = Bid OFI - Ask OFI (positive = buying pressure)
        let ofi = bid_ofi - ask_ofi;
        
        if update.is_top_of_book {
            self.ofi_level1_acc += ofi;
        }
        
        // Aggregate OFI includes all levels with weighting
        let weight = if update.level < self.ofi_levels {
            1.0 / (update.level + 1) as f64
        } else {
            0.0
        };
        self.ofi_aggregate_acc += ofi * weight;
    }

    /// Calculate OFI for one side of the book
    fn calculate_side_ofi(&self, prev_price: f64, prev_size: u64, curr_price: f64, curr_size: u64) -> f64 {
        if prev_price == 0.0 {
            return 0.0;
        }

        let prev_size = prev_size as f64;
        let curr_size = curr_size as f64;

        if curr_price > prev_price {
            // Price improved -> strong signal
            curr_size
        } else if curr_price < prev_price {
            // Price worsened
            -prev_size
        } else {
            // Price unchanged, size changed
            curr_size - prev_size
        }
    }

    /// Finalize and return accumulated features, then reset
    pub fn finalize(&mut self, book: &LimitOrderBook) -> BarFeatures {
        let total_volume = self.buy_volume_acc + self.sell_volume_acc;
        
        let trade_imbalance = if total_volume > 0 {
            (self.buy_volume_acc as f64 - self.sell_volume_acc as f64) 
                / total_volume as f64
        } else {
            0.0
        };

        let vwap = if self.volume_sum > 0 {
            self.price_volume_sum / self.volume_sum as f64
        } else {
            self.close_price
        };

        let spread_avg = if self.spread_count > 0 {
            self.spread_sum / self.spread_count as f64
        } else {
            book.spread()
        };

        let depth_ratio = if self.depth_ratio_count > 0 {
            self.depth_ratio_sum / self.depth_ratio_count as f64
        } else {
            book.depth_ratio(self.ofi_levels)
        };

        // Handle edge cases for OHLC
        let high = if self.high_price == f64::MIN { self.close_price } else { self.high_price };
        let low = if self.low_price == f64::MAX { self.close_price } else { self.low_price };

        let features = BarFeatures {
            timestamp_start: self.first_timestamp,
            timestamp_end: self.last_timestamp,
            open: self.open_price,
            high,
            low,
            close: self.close_price,
            volume: self.volume_sum,
            trade_count: self.trade_count,
            ofi_level1: self.ofi_level1_acc,
            ofi_aggregate: self.ofi_aggregate_acc,
            buy_volume: self.buy_volume_acc,
            sell_volume: self.sell_volume_acc,
            trade_imbalance,
            depth_ratio,
            spread_avg,
            spread_max: self.spread_max,
            sweep_count: self.sweep_count,
            aggressive_buy_count: self.aggressive_buy_count,
            aggressive_sell_count: self.aggressive_sell_count,
            vwap,
            vwap_deviation: if vwap > 0.0 { (self.close_price - vwap) / vwap * 100.0 } else { 0.0 },
            price_delta: self.close_price - self.open_price,
            high_low_range: high - low,
            bid_close: book.best_bid,
            ask_close: book.best_ask,
            mid_close: book.mid_price(),
        };

        // Reset accumulators
        self.reset();

        features
    }

    /// Reset all accumulators for a new bar
    pub fn reset(&mut self) {
        self.ofi_level1_acc = 0.0;
        self.ofi_aggregate_acc = 0.0;
        self.buy_volume_acc = 0;
        self.sell_volume_acc = 0;
        self.spread_sum = 0.0;
        self.spread_count = 0;
        self.spread_max = 0.0;
        self.depth_ratio_sum = 0.0;
        self.depth_ratio_count = 0;
        self.sweep_count = 0;
        self.aggressive_buy_count = 0;
        self.aggressive_sell_count = 0;
        self.price_volume_sum = 0.0;
        self.volume_sum = 0;
        self.open_price = 0.0;
        self.high_price = f64::MIN;
        self.low_price = f64::MAX;
        self.close_price = 0.0;
        self.trade_count = 0;
        self.first_timestamp = 0;
        self.last_timestamp = 0;
        self.recent_l2_removes.clear();
    }

    /// Partially finalize with remaining volume (for strict split)
    pub fn partial_finalize(&mut self, used_volume_fraction: f64, book: &LimitOrderBook) -> BarFeatures {
        // Scale volume-based accumulators by the fraction used
        let original_buy = self.buy_volume_acc;
        let original_sell = self.sell_volume_acc;
        
        self.buy_volume_acc = (original_buy as f64 * used_volume_fraction) as u64;
        self.sell_volume_acc = (original_sell as f64 * used_volume_fraction) as u64;
        
        let features = self.finalize(book);
        
        // Keep remaining for next bar
        self.buy_volume_acc = original_buy - self.buy_volume_acc;
        self.sell_volume_acc = original_sell - self.sell_volume_acc;
        
        features
    }
}
