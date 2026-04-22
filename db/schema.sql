CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- MEDALLION TABLES
CREATE TABLE IF NOT EXISTS bronze_ohlcv_raw (
  id BIGSERIAL PRIMARY KEY,
  commodity TEXT,
  ts_raw TEXT,
  open_raw TEXT,
  high_raw TEXT,
  low_raw TEXT,
  close_raw TEXT,
  volume_raw TEXT,
  source_file TEXT,
  ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS silver_ohlcv_clean (
  id BIGSERIAL PRIMARY KEY,
  commodity TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  open DOUBLE PRECISION NOT NULL,
  high DOUBLE PRECISION NOT NULL,
  low DOUBLE PRECISION NOT NULL,
  close DOUBLE PRECISION NOT NULL,
  volume DOUBLE PRECISION NOT NULL,
  UNIQUE (commodity, ts)
);

CREATE TABLE IF NOT EXISTS gold_ohlcv_features (
  id BIGSERIAL PRIMARY KEY,
  commodity TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  close DOUBLE PRECISION NOT NULL,
  ret_1 DOUBLE PRECISION,
  ret_3 DOUBLE PRECISION,
  ret_6 DOUBLE PRECISION,
  volatility_12 DOUBLE PRECISION,
  sma_6 DOUBLE PRECISION,
  sma_12 DOUBLE PRECISION,
  volume_z DOUBLE PRECISION,
  UNIQUE (commodity, ts)
);

-- CORE APP TABLES
CREATE TABLE IF NOT EXISTS signals (
  id BIGSERIAL PRIMARY KEY,
  commodity TEXT NOT NULL,
  side TEXT NOT NULL,
  entry DOUBLE PRECISION NOT NULL,
  stop_loss DOUBLE PRECISION NOT NULL,
  take_profit DOUBLE PRECISION NOT NULL,
  risk_tag TEXT NOT NULL,
  reason TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS positions (
  id UUID PRIMARY KEY,
  commodity TEXT NOT NULL,
  side TEXT NOT NULL,
  entry_price DOUBLE PRECISION NOT NULL,
  exit_price DOUBLE PRECISION,
  pnl DOUBLE PRECISION,
  status TEXT NOT NULL,
  order_created_at TIMESTAMPTZ NOT NULL,
  order_finished_at TIMESTAMPTZ
);

DROP VIEW IF EXISTS pnl_history;
CREATE VIEW pnl_history AS
SELECT id, commodity, side, entry_price, exit_price, pnl, order_created_at, order_finished_at
FROM positions
WHERE status = 'CLOSED';
