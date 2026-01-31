//! Volume Bar construction with strict split logic

use crate::features::{BarFeatures, FeatureCalculator};
use crate::order_book::LimitOrderBook;
use crate::parser::Record;

/// Volume bar builder with strict split
#[derive(Debug)]
pub struct VolumeBarBuilder {
    /// Target volume per bar
    target_volume: u64,
    
    /// Use strict splitting (split trades across bars)
    strict_split: bool,
    
    /// Current accumulated volume
    current_volume: u64,
    
    /// Bar counter
    bar_index: u64,
}

impl VolumeBarBuilder {
    pub fn new(target_volume: u64, strict_split: bool) -> Self {
        Self {
            target_volume,
            strict_split,
            current_volume: 0,
            bar_index: 0,
        }
    }

    /// Add volume from a trade and return completed bars
    pub fn add_volume(
        &mut self,
        trade_volume: u64,
        _record: &Record,
        book: &LimitOrderBook,
        feature_calc: &mut FeatureCalculator,
    ) -> Vec<BarFeatures> {
        let mut completed_bars = Vec::new();
        let mut remaining_volume = trade_volume;

        while remaining_volume > 0 {
            let volume_to_fill = self.target_volume - self.current_volume;

            if remaining_volume >= volume_to_fill {
                // This trade completes the current bar (possibly with overflow)
                self.current_volume = self.target_volume;
                remaining_volume -= volume_to_fill;

                // Calculate fraction of trade used for this bar
                let fraction_used = if self.strict_split && trade_volume > 0 {
                    volume_to_fill as f64 / trade_volume as f64
                } else {
                    1.0
                };

                // Finalize bar
                let mut bar = if self.strict_split && fraction_used < 1.0 {
                    feature_calc.partial_finalize(fraction_used, book)
                } else {
                    feature_calc.finalize(book)
                };
                
                // Set bar metadata
                bar.volume = self.target_volume;
                self.bar_index += 1;
                
                completed_bars.push(bar);

                // Reset for next bar
                self.current_volume = 0;

                // If there's remaining volume and strict split, continue to next bar
                if !self.strict_split && remaining_volume > 0 {
                    // Without strict split, put remainder in next bar
                    self.current_volume = remaining_volume;
                    remaining_volume = 0;
                }
            } else {
                // Trade doesn't complete the bar
                self.current_volume += remaining_volume;
                remaining_volume = 0;
            }
        }

        completed_bars
    }

    /// Get current bar progress
    #[allow(dead_code)]
    pub fn current_volume(&self) -> u64 {
        self.current_volume
    }

    /// Get total bars created
    #[allow(dead_code)]
    pub fn bar_count(&self) -> u64 {
        self.bar_index
    }

    /// Get fill percentage of current bar
    #[allow(dead_code)]
    pub fn fill_percentage(&self) -> f64 {
        self.current_volume as f64 / self.target_volume as f64 * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strict_split() {
        let mut builder = VolumeBarBuilder::new(500, true);
        let book = LimitOrderBook::new(10);
        let mut feature_calc = FeatureCalculator::new(5);

        // Create a dummy record
        let record = crate::parser::parse_record("L1;2;20241227060000;0;21948,25;1").unwrap();

        // Accumulate some volume
        builder.current_volume = 450;

        // Add trade of 100 - should complete bar with 50 remaining
        let bars = builder.add_volume(100, &record, &book, &mut feature_calc);
        
        assert_eq!(bars.len(), 1);
        assert_eq!(builder.current_volume, 50);
    }

    #[test]
    fn test_large_trade_multiple_bars() {
        let mut builder = VolumeBarBuilder::new(100, true);
        let book = LimitOrderBook::new(10);
        let mut feature_calc = FeatureCalculator::new(5);

        let record = crate::parser::parse_record("L1;2;20241227060000;0;21948,25;1").unwrap();

        // Large trade that creates multiple bars
        let bars = builder.add_volume(350, &record, &book, &mut feature_calc);
        
        assert_eq!(bars.len(), 3);
        assert_eq!(builder.current_volume, 50);
    }
}
