use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Ohlcv {
    pub commodity: String,
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Signal {
    pub commodity: String,
    pub side: String,
    pub entry: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub risk_tag: String,
    pub reason: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, Clone, FromRow)]
pub struct Position {
    pub id: Uuid,
    pub commodity: String,
    pub side: String,
    pub entry_price: f64,
    pub exit_price: Option<f64>,
    pub pnl: Option<f64>,
    pub status: String,
    pub order_created_at: DateTime<Utc>,
    pub order_finished_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreatePositionRequest {
    pub commodity: String,
    pub side: String,
    pub entry_price: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExitPositionRequest {
    pub exit_price: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NewsItem {
    pub title: String,
    pub summary: String,
    pub source: String,
    pub published_at: DateTime<Utc>,
    pub validated: bool,
}
