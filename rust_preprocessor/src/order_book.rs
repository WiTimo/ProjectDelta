//! Limit Order Book (LOB) reconstruction and maintenance

use crate::parser::{MarketDataType, Operation, Record};
use std::collections::BTreeMap;

/// A single price level in the order book
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct PriceLevel {
    pub price: f64,
    pub volume: u64,
    pub order_count: u32,
}

/// Side of the book
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Bid,
    Ask,
}

/// Update information returned when the book changes
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct BookUpdate {
    pub side: Option<Side>,
    pub operation: Option<Operation>,
    pub level: usize,
    pub old_price: f64,
    pub new_price: f64,
    pub old_volume: u64,
    pub new_volume: u64,
    pub is_top_of_book: bool,
}

/// Limit Order Book with best N levels on each side
#[derive(Debug, Clone)]
pub struct LimitOrderBook {
    /// Maximum depth levels to track
    max_levels: usize,
    
    /// Bid levels (buy orders) - sorted by price descending (best bid first)
    /// Key: negative price for descending order
    bids: BTreeMap<i64, PriceLevel>,
    
    /// Ask levels (sell orders) - sorted by price ascending (best ask first)
    asks: BTreeMap<i64, PriceLevel>,
    
    /// Best bid/ask from L1 data (authoritative)
    pub best_bid: f64,
    pub best_bid_size: u64,
    pub best_ask: f64,
    pub best_ask_size: u64,
    
    /// Previous best bid/ask for OFI calculation
    pub prev_best_bid: f64,
    pub prev_best_bid_size: u64,
    pub prev_best_ask: f64,
    pub prev_best_ask_size: u64,
    
    /// Last trade info
    pub last_price: f64,
    pub last_volume: u64,
}

impl LimitOrderBook {
    pub fn new(max_levels: usize) -> Self {
        Self {
            max_levels,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            best_bid: 0.0,
            best_bid_size: 0,
            best_ask: 0.0,
            best_ask_size: 0,
            prev_best_bid: 0.0,
            prev_best_bid_size: 0,
            prev_best_ask: 0.0,
            prev_best_ask_size: 0,
            last_price: 0.0,
            last_volume: 0,
        }
    }

    /// Convert price to integer key (preserves ordering, handles ticks of 0.25)
    fn price_to_key(price: f64) -> i64 {
        (price * 10000.0) as i64
    }

    #[allow(dead_code)]
    fn key_to_price(key: i64) -> f64 {
        (key as f64) / 10000.0
    }

    /// Update the order book with a new record
    pub fn update(&mut self, record: &Record) -> BookUpdate {
        match record {
            Record::L1 { data_type, price, volume, .. } => {
                self.update_l1(*data_type, *price, *volume)
            }
            Record::L2 { data_type, operation, position, price, volume, .. } => {
                self.update_l2(*data_type, *operation, *position, *price, *volume)
            }
        }
    }

    fn update_l1(&mut self, data_type: MarketDataType, price: f64, volume: u64) -> BookUpdate {
        let mut update = BookUpdate::default();

        match data_type {
            MarketDataType::Bid => {
                // Store previous values for OFI
                self.prev_best_bid = self.best_bid;
                self.prev_best_bid_size = self.best_bid_size;
                
                update.side = Some(Side::Bid);
                update.old_price = self.best_bid;
                update.old_volume = self.best_bid_size;
                update.new_price = price;
                update.new_volume = volume;
                update.is_top_of_book = true;
                update.level = 0;

                self.best_bid = price;
                self.best_bid_size = volume;
            }
            MarketDataType::Ask => {
                // Store previous values for OFI
                self.prev_best_ask = self.best_ask;
                self.prev_best_ask_size = self.best_ask_size;
                
                update.side = Some(Side::Ask);
                update.old_price = self.best_ask;
                update.old_volume = self.best_ask_size;
                update.new_price = price;
                update.new_volume = volume;
                update.is_top_of_book = true;
                update.level = 0;

                self.best_ask = price;
                self.best_ask_size = volume;
            }
            MarketDataType::Last => {
                self.last_price = price;
                self.last_volume = volume;
            }
            _ => {}
        }

        update
    }

    fn update_l2(
        &mut self,
        data_type: MarketDataType,
        operation: Operation,
        position: usize,
        price: f64,
        volume: u64,
    ) -> BookUpdate {
        let mut update = BookUpdate::default();
        update.operation = Some(operation);
        update.level = position;
        update.new_price = price;
        update.new_volume = volume;
        update.is_top_of_book = position == 0;

        let key = Self::price_to_key(price);

        match data_type {
            MarketDataType::Bid => {
                update.side = Some(Side::Bid);
                let neg_key = -key; // Negative for descending order

                match operation {
                    Operation::Add | Operation::Update => {
                        if let Some(level) = self.bids.get(&neg_key) {
                            update.old_price = level.price;
                            update.old_volume = level.volume;
                        }
                        self.bids.insert(neg_key, PriceLevel {
                            price,
                            volume,
                            order_count: 1,
                        });
                        // Trim to max levels
                        while self.bids.len() > self.max_levels {
                            self.bids.pop_last();
                        }
                    }
                    Operation::Remove => {
                        if let Some(level) = self.bids.remove(&neg_key) {
                            update.old_price = level.price;
                            update.old_volume = level.volume;
                        }
                    }
                }
            }
            MarketDataType::Ask => {
                update.side = Some(Side::Ask);

                match operation {
                    Operation::Add | Operation::Update => {
                        if let Some(level) = self.asks.get(&key) {
                            update.old_price = level.price;
                            update.old_volume = level.volume;
                        }
                        self.asks.insert(key, PriceLevel {
                            price,
                            volume,
                            order_count: 1,
                        });
                        // Trim to max levels
                        while self.asks.len() > self.max_levels {
                            self.asks.pop_last();
                        }
                    }
                    Operation::Remove => {
                        if let Some(level) = self.asks.remove(&key) {
                            update.old_price = level.price;
                            update.old_volume = level.volume;
                        }
                    }
                }
            }
            _ => {}
        }

        update
    }

    /// Get bid levels (best first)
    #[allow(dead_code)]
    pub fn get_bids(&self, n: usize) -> Vec<&PriceLevel> {
        self.bids.values().take(n).collect()
    }

    /// Get ask levels (best first)
    #[allow(dead_code)]
    pub fn get_asks(&self, n: usize) -> Vec<&PriceLevel> {
        self.asks.values().take(n).collect()
    }

    /// Current spread
    pub fn spread(&self) -> f64 {
        if self.best_ask > 0.0 && self.best_bid > 0.0 {
            self.best_ask - self.best_bid
        } else {
            0.0
        }
    }

    /// Mid price
    pub fn mid_price(&self) -> f64 {
        if self.best_ask > 0.0 && self.best_bid > 0.0 {
            (self.best_ask + self.best_bid) / 2.0
        } else if self.best_bid > 0.0 {
            self.best_bid
        } else if self.best_ask > 0.0 {
            self.best_ask
        } else {
            self.last_price
        }
    }

    /// Total bid depth (weighted by distance)
    pub fn bid_depth(&self, levels: usize) -> f64 {
        let mid = self.mid_price();
        if mid <= 0.0 {
            return 0.0;
        }

        self.bids
            .values()
            .take(levels)
            .map(|level| {
                let distance = mid - level.price;
                let weight = if distance > 0.0 { 1.0 / (1.0 + distance) } else { 1.0 };
                level.volume as f64 * weight
            })
            .sum()
    }

    /// Total ask depth (weighted by distance)
    pub fn ask_depth(&self, levels: usize) -> f64 {
        let mid = self.mid_price();
        if mid <= 0.0 {
            return 0.0;
        }

        self.asks
            .values()
            .take(levels)
            .map(|level| {
                let distance = level.price - mid;
                let weight = if distance > 0.0 { 1.0 / (1.0 + distance) } else { 1.0 };
                level.volume as f64 * weight
            })
            .sum()
    }

    /// Depth ratio (bid_depth / ask_depth)
    pub fn depth_ratio(&self, levels: usize) -> f64 {
        let ask_depth = self.ask_depth(levels);
        if ask_depth > 0.0 {
            self.bid_depth(levels) / ask_depth
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lob_basic() {
        let mut lob = LimitOrderBook::new(10);
        
        // Simulate L1 updates
        lob.update_l1(MarketDataType::Bid, 21947.0, 5);
        lob.update_l1(MarketDataType::Ask, 21948.25, 3);
        
        assert!((lob.best_bid - 21947.0).abs() < 0.001);
        assert!((lob.best_ask - 21948.25).abs() < 0.001);
        assert!((lob.spread() - 1.25).abs() < 0.001);
        assert!((lob.mid_price() - 21947.625).abs() < 0.001);
    }

    #[test]
    fn test_depth_ratio() {
        let mut lob = LimitOrderBook::new(10);
        
        lob.update_l1(MarketDataType::Bid, 100.0, 10);
        lob.update_l1(MarketDataType::Ask, 101.0, 10);
        
        // Equal depth should give ratio close to 1
        let ratio = lob.depth_ratio(5);
        assert!((ratio - 1.0).abs() < 0.5);
    }
}
