-- CRUDEMCX | Full Medallion Schema
-- Run: psql -U postgres -d crudemcx -f db/schema_medallion.sql

CREATE SCHEMA IF NOT EXISTS bronze;
CREATE TABLE IF NOT EXISTS bronze.ohlcv_raw (
    row_id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(30) NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    open NUMERIC(14,4), high NUMERIC(14,4), low NUMERIC(14,4), close NUMERIC(14,4),
    volume BIGINT,
    source_system VARCHAR(60) NOT NULL,
    batch_id UUID NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    quality_flags TEXT[] NOT NULL DEFAULT '{}',
    is_duplicate BOOLEAN NOT NULL DEFAULT FALSE,
    is_quarantined BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (ticker, ts)
);
CREATE INDEX IF NOT EXISTS idx_bronze_ticker_ts ON bronze.ohlcv_raw (ticker, ts DESC);
CREATE INDEX IF NOT EXISTS idx_bronze_batch ON bronze.ohlcv_raw (batch_id);
CREATE INDEX IF NOT EXISTS idx_bronze_quarantine ON bronze.ohlcv_raw (is_quarantined) WHERE is_quarantined = TRUE;

CREATE TABLE IF NOT EXISTS bronze.ingest_audit (
    audit_id BIGSERIAL PRIMARY KEY,
    batch_id UUID NOT NULL,
    ticker VARCHAR(30),
    source_system VARCHAR(60),
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    rows_in_file INT,
    rows_deduped INT,
    rows_inserted INT,
    rows_updated INT,
    rows_flagged INT,
    rows_quarantined INT,
    status VARCHAR(20) DEFAULT 'RUNNING',
    error_message TEXT
);

CREATE SCHEMA IF NOT EXISTS silver;
CREATE TABLE IF NOT EXISTS silver.ohlcv_clean (
    row_id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(30) NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    open NUMERIC(14,4) NOT NULL, high NUMERIC(14,4) NOT NULL, low NUMERIC(14,4) NOT NULL, close NUMERIC(14,4) NOT NULL,
    volume BIGINT NOT NULL,
    bronze_batch_id UUID,
    source_system VARCHAR(60),
    was_repaired BOOLEAN NOT NULL DEFAULT FALSE,
    was_gap_filled BOOLEAN NOT NULL DEFAULT FALSE,
    repair_summary TEXT,
    silver_batch_id UUID NOT NULL,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (ticker, ts)
);
CREATE INDEX IF NOT EXISTS idx_silver_ticker_ts ON silver.ohlcv_clean (ticker, ts DESC);
CREATE INDEX IF NOT EXISTS idx_silver_repaired ON silver.ohlcv_clean (was_repaired) WHERE was_repaired = TRUE;

CREATE TABLE IF NOT EXISTS silver.repair_log (
    log_id BIGSERIAL PRIMARY KEY,
    silver_batch_id UUID NOT NULL,
    ticker VARCHAR(30) NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    field VARCHAR(20) NOT NULL,
    issue VARCHAR(60) NOT NULL,
    strategy VARCHAR(60) NOT NULL,
    original_value NUMERIC(18,6),
    repaired_value NUMERIC(18,6),
    logged_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_repair_batch ON silver.repair_log (silver_batch_id);

CREATE TABLE IF NOT EXISTS silver.transform_audit (
    audit_id BIGSERIAL PRIMARY KEY,
    silver_batch_id UUID NOT NULL,
    bronze_batch_id UUID,
    ticker VARCHAR(30),
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    rows_read_bronze INT,
    rows_quarantined INT,
    rows_repaired INT,
    rows_gap_filled INT,
    rows_written INT,
    status VARCHAR(20) DEFAULT 'RUNNING',
    error_message TEXT
);

CREATE SCHEMA IF NOT EXISTS gold;
CREATE TABLE IF NOT EXISTS gold.ml_features (
    row_id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(30) NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    close NUMERIC(14,4),
    ret_1h NUMERIC(12,8), ret_4h NUMERIC(12,8), ret_8h NUMERIC(12,8), ret_24h NUMERIC(12,8),
    log_ret_1h NUMERIC(12,8),
    ema_9 NUMERIC(14,4), ema_21 NUMERIC(14,4), ema_50 NUMERIC(14,4),
    ema_9_21_cross NUMERIC(12,8), ema_21_50_cross NUMERIC(12,8),
    rsi_7 NUMERIC(8,4), rsi_14 NUMERIC(8,4),
    macd NUMERIC(12,8), macd_signal NUMERIC(12,8), macd_hist NUMERIC(12,8),
    roc_5 NUMERIC(12,8), roc_10 NUMERIC(12,8),
    atr_14 NUMERIC(12,4), atr_pct NUMERIC(12,8),
    bb_upper NUMERIC(14,4), bb_lower NUMERIC(14,4), bb_pct_b NUMERIC(12,8), bb_width NUMERIC(12,8),
    rolling_std_24h NUMERIC(12,8),
    log_volume NUMERIC(12,8), vol_ma_ratio NUMERIC(12,8), obv_norm NUMERIC(12,8),
    hour_sin NUMERIC(10,8), hour_cos NUMERIC(10,8), dow_sin NUMERIC(10,8), dow_cos NUMERIC(10,8),
    target_4h_ret NUMERIC(12,8), target_8h_ret NUMERIC(12,8), target_24h_ret NUMERIC(12,8),
    target_4h_dir SMALLINT, target_8h_dir SMALLINT, target_24h_dir SMALLINT,
    split VARCHAR(10),
    was_gap_filled BOOLEAN NOT NULL DEFAULT FALSE,
    gold_batch_id UUID NOT NULL,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (ticker, ts)
);
CREATE INDEX IF NOT EXISTS idx_gold_ticker_ts ON gold.ml_features (ticker, ts DESC);
CREATE INDEX IF NOT EXISTS idx_gold_split ON gold.ml_features (ticker, split);

CREATE TABLE IF NOT EXISTS gold.feature_audit (
    audit_id BIGSERIAL PRIMARY KEY,
    gold_batch_id UUID NOT NULL,
    ticker VARCHAR(30),
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    rows_read INT,
    rows_written INT,
    rows_dropped INT,
    n_features INT,
    train_rows INT,
    val_rows INT,
    test_rows INT,
    status VARCHAR(20) DEFAULT 'RUNNING',
    error_message TEXT
);
