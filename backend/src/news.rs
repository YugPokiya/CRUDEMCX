use crate::models::NewsItem;
use anyhow::Result;
use chrono::Utc;

pub async fn fetch_hourly_news() -> Result<Vec<NewsItem>> {
    let sample = vec![NewsItem {
        title: "Crude oil futures react to inventory draw".to_string(),
        summary: "Inventory data indicates a tighter near-term oil balance.".to_string(),
        source: "EnergyWire".to_string(),
        published_at: Utc::now(),
        validated: true,
    }];

    Ok(sample)
}

pub fn infer_sentiment_action(headline: &str) -> &'static str {
    let lower = headline.to_lowercase();
    if lower.contains("draw") || lower.contains("supply cut") {
        "BUY"
    } else if lower.contains("surplus") || lower.contains("output increase") {
        "SELL"
    } else {
        "HOLD"
    }
}
