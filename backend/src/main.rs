mod db;
mod models;
mod news;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use models::{CreatePositionRequest, ExitPositionRequest, Ohlcv, Signal};
use sqlx::postgres::PgPoolOptions;
use sqlx::{Pool, Postgres};
use std::env;
use tracing::info;
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    pool: Pool<Postgres>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").compact().init();

    let database_url = env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5432/mcx_trade".to_string());

    let pool = PgPoolOptions::new().max_connections(5).connect(&database_url).await?;
    let app_state = AppState { pool };

    let app = Router::new()
        .route("/health", get(health))
        .route("/api/ohlcv/:commodity", get(get_ohlcv))
        .route("/api/signals/:commodity", get(get_signal))
        .route("/api/news", get(get_news))
        .route("/api/positions", get(list_positions).post(open_position))
        .route("/api/positions/:id/exit", post(close_position))
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    info!("Backend listening on 0.0.0.0:8080");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> &'static str { "ok" }

async fn get_ohlcv(Path(commodity): Path<String>) -> Json<Vec<Ohlcv>> {
    let now = Utc::now();
    let mut candles = Vec::new();
    for i in 0..30 {
        let base = 7000.0 + (i as f64 * 2.5);
        candles.push(Ohlcv {
            commodity: commodity.clone(),
            timestamp: now - chrono::Duration::hours(i),
            open: base - 5.0,
            high: base + 9.0,
            low: base - 12.0,
            close: base,
            volume: 1000.0 + (i as f64 * 31.0),
        });
    }
    Json(candles)
}

async fn get_signal(Path(commodity): Path<String>, State(state): State<AppState>) -> Result<Json<Signal>, (StatusCode, String)> {
    let signal = Signal {
        commodity: commodity.clone(),
        side: "BUY".to_string(),
        entry: 7124.5,
        stop_loss: 7060.0,
        take_profit: 7240.0,
        risk_tag: "Safety".to_string(),
        reason: "LSTM score bullish with supportive validated news momentum.".to_string(),
        created_at: Utc::now(),
    };
    db::save_signal(&state.pool, &signal).await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(signal))
}

async fn get_news() -> Result<Json<Vec<serde_json::Value>>, (StatusCode, String)> {
    let items = news::fetch_hourly_news().await.map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
    let enriched: Vec<_> = items.into_iter().map(|n| serde_json::json!({
        "title": n.title,
        "summary": n.summary,
        "source": n.source,
        "published_at": n.published_at,
        "validated": n.validated,
        "action_hint": news::infer_sentiment_action(&n.title)
    })).collect();
    Ok(Json(enriched))
}

async fn list_positions(State(state): State<AppState>) -> Result<Json<Vec<models::Position>>, (StatusCode, String)> {
    let rows = db::list_positions(&state.pool).await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(rows))
}

async fn open_position(State(state): State<AppState>, Json(req): Json<CreatePositionRequest>) -> Result<Json<models::Position>, (StatusCode, String)> {
    let position = db::create_position(&state.pool, req).await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(position))
}

async fn close_position(Path(id): Path<Uuid>, State(state): State<AppState>, Json(req): Json<ExitPositionRequest>) -> Result<Json<models::Position>, (StatusCode, String)> {
    let position = db::exit_position(&state.pool, id, req).await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(position))
}
