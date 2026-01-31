//! Parser for NinjaTrader L1/L2 tick data

use anyhow::{anyhow, Result};
use chrono::NaiveDateTime;

/// Market data types from NinjaTrader
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketDataType {
    Ask = 0,
    Bid = 1,
    Last = 2,
    DailyHigh = 3,
    DailyLow = 4,
    DailyVolume = 5,
    LastClose = 6,
    Opening = 7,
    OpenInterest = 8,
    Settlement = 9,
    Unknown = 10,
}

impl MarketDataType {
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => MarketDataType::Ask,
            1 => MarketDataType::Bid,
            2 => MarketDataType::Last,
            3 => MarketDataType::DailyHigh,
            4 => MarketDataType::DailyLow,
            5 => MarketDataType::DailyVolume,
            6 => MarketDataType::LastClose,
            7 => MarketDataType::Opening,
            8 => MarketDataType::OpenInterest,
            9 => MarketDataType::Settlement,
            _ => MarketDataType::Unknown,
        }
    }
}

/// L2 Operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Add = 0,
    Update = 1,
    Remove = 2,
}

impl Operation {
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => Operation::Add,
            1 => Operation::Update,
            2 => Operation::Remove,
            _ => Operation::Update, // Default to update
        }
    }
}

/// Parsed record from tick data
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum Record {
    L1 {
        data_type: MarketDataType,
        timestamp: NaiveDateTime,
        timestamp_offset_100ns: i64,
        price: f64,
        volume: u64,
    },
    L2 {
        data_type: MarketDataType,
        timestamp: NaiveDateTime,
        timestamp_offset_100ns: i64,
        operation: Operation,
        position: usize,
        market_maker: String,
        price: f64,
        volume: u64,
    },
}

#[allow(dead_code)]
impl Record {
    pub fn timestamp(&self) -> NaiveDateTime {
        match self {
            Record::L1 { timestamp, .. } => *timestamp,
            Record::L2 { timestamp, .. } => *timestamp,
        }
    }

    pub fn timestamp_nanos(&self) -> i64 {
        match self {
            Record::L1 { timestamp, timestamp_offset_100ns, .. } => {
                timestamp.and_utc().timestamp_nanos_opt().unwrap_or(0) + timestamp_offset_100ns * 100
            }
            Record::L2 { timestamp, timestamp_offset_100ns, .. } => {
                timestamp.and_utc().timestamp_nanos_opt().unwrap_or(0) + timestamp_offset_100ns * 100
            }
        }
    }

    pub fn price(&self) -> f64 {
        match self {
            Record::L1 { price, .. } => *price,
            Record::L2 { price, .. } => *price,
        }
    }
}

/// Parse a single line from the tick data file
pub fn parse_record(line: &str) -> Result<Record> {
    let parts: Vec<&str> = line.split(';').collect();
    
    if parts.is_empty() {
        return Err(anyhow!("Empty line"));
    }

    match parts[0] {
        "L1" => parse_l1(&parts),
        "L2" => parse_l2(&parts),
        _ => Err(anyhow!("Unknown record type: {}", parts[0])),
    }
}

fn parse_l1(parts: &[&str]) -> Result<Record> {
    // L1;data_type;timestamp;offset;price;volume
    if parts.len() < 6 {
        return Err(anyhow!("L1 record too short: {} parts", parts.len()));
    }

    let data_type = MarketDataType::from_u8(
        parts[1].parse::<u8>().map_err(|e| anyhow!("Invalid data type: {}", e))?
    );
    
    let timestamp = parse_timestamp(parts[2])?;
    let timestamp_offset_100ns = parts[3].parse::<i64>()
        .map_err(|e| anyhow!("Invalid offset: {}", e))?;
    
    let price = parse_price(parts[4])?;
    let volume = parts[5].parse::<u64>()
        .map_err(|e| anyhow!("Invalid volume: {}", e))?;

    Ok(Record::L1 {
        data_type,
        timestamp,
        timestamp_offset_100ns,
        price,
        volume,
    })
}

fn parse_l2(parts: &[&str]) -> Result<Record> {
    // L2;data_type;timestamp;offset;operation;position;market_maker;price;volume
    if parts.len() < 9 {
        return Err(anyhow!("L2 record too short: {} parts", parts.len()));
    }

    let data_type = MarketDataType::from_u8(
        parts[1].parse::<u8>().map_err(|e| anyhow!("Invalid data type: {}", e))?
    );
    
    let timestamp = parse_timestamp(parts[2])?;
    let timestamp_offset_100ns = parts[3].parse::<i64>()
        .map_err(|e| anyhow!("Invalid offset: {}", e))?;
    
    let operation = Operation::from_u8(
        parts[4].parse::<u8>().map_err(|e| anyhow!("Invalid operation: {}", e))?
    );
    
    let position = parts[5].parse::<usize>()
        .map_err(|e| anyhow!("Invalid position: {}", e))?;
    
    let market_maker = parts[6].to_string();
    
    let price = parse_price(parts[7])?;
    let volume = parts[8].parse::<u64>()
        .map_err(|e| anyhow!("Invalid volume: {}", e))?;

    Ok(Record::L2 {
        data_type,
        timestamp,
        timestamp_offset_100ns,
        operation,
        position,
        market_maker,
        price,
        volume,
    })
}

/// Parse NinjaTrader timestamp format: YYYYMMDDhhmmss
fn parse_timestamp(s: &str) -> Result<NaiveDateTime> {
    if s.len() != 14 {
        return Err(anyhow!("Invalid timestamp length: {}", s.len()));
    }

    NaiveDateTime::parse_from_str(s, "%Y%m%d%H%M%S")
        .map_err(|e| anyhow!("Invalid timestamp format: {}", e))
}

/// Parse price with European decimal format (comma as decimal separator)
fn parse_price(s: &str) -> Result<f64> {
    // Replace comma with dot for parsing
    let normalized = s.replace(',', ".");
    let price = normalized.parse::<f64>()
        .map_err(|e| anyhow!("Invalid price '{}': {}", s, e))?;
    
    // Validate price is positive and reasonable for NQ futures
    if price <= 0.0 {
        return Err(anyhow!("Invalid zero/negative price: {}", price));
    }
    
    // NQ futures should be in a reasonable range (10k-50k as of 2024-2026)
    // This catches obviously bad data
    if price < 1000.0 || price > 100000.0 {
        return Err(anyhow!("Price {} outside valid NQ range (1000-100000)", price));
    }
    
    Ok(price)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_l1() {
        let line = "L1;0;20241227060000;3240000;21948,25;1";
        let record = parse_record(line).unwrap();
        
        if let Record::L1 { data_type, price, volume, .. } = record {
            assert_eq!(data_type, MarketDataType::Ask);
            assert!((price - 21948.25).abs() < 0.001);
            assert_eq!(volume, 1);
        } else {
            panic!("Expected L1 record");
        }
    }

    #[test]
    fn test_parse_l2() {
        let line = "L2;1;20241227060000;3240000;2;0;;21947,5;0";
        let record = parse_record(line).unwrap();
        
        if let Record::L2 { data_type, operation, position, price, volume, .. } = record {
            assert_eq!(data_type, MarketDataType::Bid);
            assert_eq!(operation, Operation::Remove);
            assert_eq!(position, 0);
            assert!((price - 21947.5).abs() < 0.001);
            assert_eq!(volume, 0);
        } else {
            panic!("Expected L2 record");
        }
    }

    #[test]
    fn test_parse_price_european() {
        assert!((parse_price("21948,25").unwrap() - 21948.25).abs() < 0.001);
        assert!((parse_price("21947,5").unwrap() - 21947.5).abs() < 0.001);
    }
}
