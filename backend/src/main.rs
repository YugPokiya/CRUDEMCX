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
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use sqlx::{ConnectOptions, Pool, Postgres};
use std::env;
use std::io::ErrorKind;
use std::net::{SocketAddr, ToSocketAddrs};
use std::str::FromStr;
use std::time::Duration;
use tracing::{info, warn};
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

    let max_connections = env::var("DB_MAX_CONNECTIONS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(5);

    let acquire_timeout_secs = env::var("DB_ACQUIRE_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(8);

    let connect_opts = PgConnectOptions::from_str(&database_url)?
        .log_statements(tracing::log::LevelFilter::Off)
        .to_owned();

    let pool = PgPoolOptions::new()
        .max_connections(max_connections)
        .min_connections(0)
        .acquire_timeout(Duration::from_secs(acquire_timeout_secs))
        .connect_lazy_with(connect_opts);

    match sqlx::query_scalar::<_, i32>("SELECT 1").fetch_one(&pool).await {
        Ok(_) => info!("Database connectivity check: OK"),
        Err(err) => warn!("Database connectivity check failed at startup: {err}. API will start; DB routes may return errors until DB is reachable."),
    }

    let app_state = AppState { pool };

    let app = Router::new()
        .route("/health", get(health))
        .route("/health/db", get(health_db))
        .route("/api/ohlcv/:commodity", get(get_ohlcv))
        .route("/api/signals/:commodity", get(get_signal))
        .route("/api/news", get(get_news))
        .route("/api/positions", get(list_positions).post(open_position))
        .route("/api/positions/:id/exit", post(close_position))
        .with_state(app_state);

    let bind_addr = env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
    let bind_retry_ports = env::var("BIND_RETRY_PORTS")
        .ok()
        .and_then(|v| v.parse::<u16>().ok())
        .unwrap_or(0);
    let (listener, resolved_addr) = bind_with_optional_retry(&bind_addr, bind_retry_ports).await?;
    info!("Backend listening on {resolved_addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn bind_with_optional_retry(
    bind_addr: &str,
    retry_ports: u16,
) -> anyhow::Result<(tokio::net::TcpListener, SocketAddr)> {
    let mut addr_iter = bind_addr.to_socket_addrs()?;
    let base = addr_iter
        .next()
        .ok_or_else(|| anyhow::anyhow!("Unable to resolve bind address: {bind_addr}"))?;

    let mut last_addr_in_use: Option<std::io::Error> = None;
    for offset in 0..=retry_ports {
        let candidate = SocketAddr::new(base.ip(), base.port().saturating_add(offset));
        match tokio::net::TcpListener::bind(candidate).await {
            Ok(listener) => return Ok((listener, candidate)),
            Err(err) if err.kind() == ErrorKind::AddrInUse => {
                warn!("Port {} already in use, trying next port", candidate.port());
                last_addr_in_use = Some(err);
            }
            Err(err) => return Err(err.into()),
        }
    }

    if let Some(err) = last_addr_in_use {
        return Err(anyhow::anyhow!(
            "Address already in use starting at {bind_addr}. \
Set BIND_ADDR to a free port or set BIND_RETRY_PORTS>0. Last error: {err}"
        ));
    }
    Err(anyhow::anyhow!("Unable to bind to any address starting at {bind_addr}"))
}

async fn health() -> &'static str {
    "ok"
}

async fn health_db(State(state): State<AppState>) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    match sqlx::query_scalar::<_, i32>("SELECT 1").fetch_one(&state.pool).await {
        Ok(_) => Ok(Json(serde_json::json!({"status": "ok"}))),
        Err(err) => Err((StatusCode::SERVICE_UNAVAILABLE, format!("db_unavailable: {err}"))),
    }
}

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

async fn get_signal(
    Path(commodity): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Signal>, (StatusCode, String)> {
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
    db::save_signal(&state.pool, &signal)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(signal))
}

async fn get_news() -> Result<Json<Vec<serde_json::Value>>, (StatusCode, String)> {
    let items = news::fetch_hourly_news()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
    let enriched: Vec<_> = items
        .into_iter()
        .map(|n| {
            serde_json::json!({
                "title": n.title,
                "summary": n.summary,
                "source": n.source,
                "published_at": n.published_at,
                "validated": n.validated,
                "action_hint": news::infer_sentiment_action(&n.title)
            })
        })
        .collect();
    Ok(Json(enriched))
}

async fn list_positions(
    State(state): State<AppState>,
) -> Result<Json<Vec<models::Position>>, (StatusCode, String)> {
    let rows = db::list_positions(&state.pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(rows))
}

async fn open_position(
    State(state): State<AppState>,
    Json(req): Json<CreatePositionRequest>,
) -> Result<Json<models::Position>, (StatusCode, String)> {
    let position = db::create_position(&state.pool, req)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(position))
}

async fn close_position(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
    Json(req): Json<ExitPositionRequest>,
) -> Result<Json<models::Position>, (StatusCode, String)> {
    let position = db::exit_position(&state.pool, id, req)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(position))
}
