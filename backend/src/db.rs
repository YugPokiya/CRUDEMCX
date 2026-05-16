use crate::models::{CreatePositionRequest, ExitPositionRequest, Position, Signal};
use anyhow::Result;
use chrono::Utc;
use sqlx::{Pool, Postgres};
use uuid::Uuid;

pub async fn list_positions(pool: &Pool<Postgres>) -> Result<Vec<Position>> {
    let rows = sqlx::query_as::<_, Position>(
        r#"
        SELECT id, commodity, side, entry_price, exit_price, pnl, status,
               order_created_at, order_finished_at
        FROM positions
        ORDER BY order_created_at DESC
        "#,
    )
    .fetch_all(pool)
    .await?;

    Ok(rows)
}

pub async fn create_position(pool: &Pool<Postgres>, req: CreatePositionRequest) -> Result<Position> {
    let id = Uuid::new_v4();
    let now = Utc::now();

    let row = sqlx::query_as::<_, Position>(
        r#"
        INSERT INTO positions (id, commodity, side, entry_price, status, order_created_at)
        VALUES ($1, $2, $3, $4, 'OPEN', $5)
        RETURNING id, commodity, side, entry_price, exit_price, pnl, status,
                  order_created_at, order_finished_at
        "#,
    )
    .bind(id)
    .bind(req.commodity)
    .bind(req.side)
    .bind(req.entry_price)
    .bind(now)
    .fetch_one(pool)
    .await?;

    Ok(row)
}

pub async fn exit_position(
    pool: &Pool<Postgres>,
    position_id: Uuid,
    req: ExitPositionRequest,
) -> Result<Position> {
    let row = sqlx::query_as::<_, Position>(
        r#"
        UPDATE positions
        SET exit_price = $2,
            pnl = CASE WHEN side = 'BUY' THEN ($2 - entry_price) ELSE (entry_price - $2) END,
            status = 'CLOSED',
            order_finished_at = $3
        WHERE id = $1
        RETURNING id, commodity, side, entry_price, exit_price, pnl, status,
                  order_created_at, order_finished_at
        "#,
    )
    .bind(position_id)
    .bind(req.exit_price)
    .bind(Utc::now())
    .fetch_one(pool)
    .await?;

    Ok(row)
}

pub async fn save_signal(pool: &Pool<Postgres>, signal: &Signal) -> Result<()> {
    sqlx::query(
        r#"
        INSERT INTO signals (commodity, side, entry, stop_loss, take_profit, risk_tag, reason, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        "#,
    )
    .bind(&signal.commodity)
    .bind(&signal.side)
    .bind(signal.entry)
    .bind(signal.stop_loss)
    .bind(signal.take_profit)
    .bind(&signal.risk_tag)
    .bind(&signal.reason)
    .bind(signal.created_at)
    .execute(pool)
    .await?;

    Ok(())
}
