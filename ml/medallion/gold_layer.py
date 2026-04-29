"""CRUDEMCX | Gold Layer (Medallion Architecture)."""

import logging
import os
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "crudemcx"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}
TICKER = os.getenv("TICKER", "NCRUDEOIL")

GOLD_DDL = """
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
"""

FEATURE_COLS = [
    "ticker", "ts", "close", "ret_1h", "ret_4h", "ret_8h", "ret_24h", "log_ret_1h",
    "ema_9", "ema_21", "ema_50", "ema_9_21_cross", "ema_21_50_cross",
    "rsi_7", "rsi_14", "macd", "macd_signal", "macd_hist", "roc_5", "roc_10",
    "atr_14", "atr_pct", "bb_upper", "bb_lower", "bb_pct_b", "bb_width", "rolling_std_24h",
    "log_volume", "vol_ma_ratio", "obv_norm", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "target_4h_ret", "target_8h_ret", "target_24h_ret", "target_4h_dir", "target_8h_dir", "target_24h_dir",
    "split", "was_gap_filled",
]


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(s: pd.Series, period: int) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    hl = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift()).abs()
    lpc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def cyclic_encode(series: pd.Series, max_val: int):
    return np.sin(2 * np.pi * series / max_val), np.cos(2 * np.pi * series / max_val)


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    t_end = int(n * 0.70)
    v_end = int(n * 0.85)
    df = df.copy()
    df["split"] = "train"
    df.iloc[t_end:v_end, df.columns.get_loc("split")] = "val"
    df.iloc[v_end:, df.columns.get_loc("split")] = "test"
    return df


def read_silver(conn, ticker: str) -> pd.DataFrame:
    sql = """
    SELECT ts, open, high, low, close, volume, was_gap_filled, ticker
    FROM silver.ohlcv_clean
    WHERE ticker = %s
    ORDER BY ts ASC;
    """
    df = pd.read_sql(sql, conn, params=(ticker,), parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def compute_gold_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["ticker"] = df["ticker"]
    f["ts"] = df["ts"]
    f["close"] = df["close"]
    f["was_gap_filled"] = df["was_gap_filled"]

    c = df["close"]
    f["ret_1h"] = c.pct_change(1)
    f["ret_4h"] = c.pct_change(4)
    f["ret_8h"] = c.pct_change(8)
    f["ret_24h"] = c.pct_change(24)
    f["log_ret_1h"] = np.log(c / c.shift(1))

    f["ema_9"] = ema(c, 9)
    f["ema_21"] = ema(c, 21)
    f["ema_50"] = ema(c, 50)
    f["ema_9_21_cross"] = (f["ema_9"] - f["ema_21"]) / f["ema_21"]
    f["ema_21_50_cross"] = (f["ema_21"] - f["ema_50"]) / f["ema_50"]

    f["rsi_7"] = rsi(c, 7)
    f["rsi_14"] = rsi(c, 14)
    macd_line = ema(c, 12) - ema(c, 26)
    macd_signal = ema(macd_line, 9)
    f["macd"] = macd_line
    f["macd_signal"] = macd_signal
    f["macd_hist"] = macd_line - macd_signal
    f["roc_5"] = c.pct_change(5)
    f["roc_10"] = c.pct_change(10)

    atr14 = atr(df, 14)
    f["atr_14"] = atr14
    f["atr_pct"] = atr14 / c
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    f["bb_upper"] = bb_mid + 2 * bb_std
    f["bb_lower"] = bb_mid - 2 * bb_std
    bb_range = f["bb_upper"] - f["bb_lower"]
    f["bb_pct_b"] = (c - f["bb_lower"]) / bb_range.replace(0, np.nan)
    f["bb_width"] = bb_range / bb_mid
    f["rolling_std_24h"] = c.rolling(24).std() / c

    vol = df["volume"].replace(0, np.nan)
    f["log_volume"] = np.log1p(vol)
    f["vol_ma_ratio"] = vol / vol.rolling(24).mean().replace(0, np.nan)
    obv_series = obv(df)
    f["obv_norm"] = (obv_series - obv_series.rolling(168).mean()) / (obv_series.rolling(168).std() + 1e-9)

    ts_ist = df["ts"].dt.tz_convert("Asia/Kolkata")
    f["hour_sin"], f["hour_cos"] = cyclic_encode(ts_ist.dt.hour, 24)
    f["dow_sin"], f["dow_cos"] = cyclic_encode(ts_ist.dt.dayofweek, 7)

    f["target_4h_ret"] = c.shift(-4) / c - 1
    f["target_8h_ret"] = c.shift(-8) / c - 1
    f["target_24h_ret"] = c.shift(-24) / c - 1
    f["target_4h_dir"] = (f["target_4h_ret"] > 0).astype("Int8")
    f["target_8h_dir"] = (f["target_8h_ret"] > 0).astype("Int8")
    f["target_24h_dir"] = (f["target_24h_ret"] > 0).astype("Int8")
    return f


def write_gold(conn, df: pd.DataFrame, gold_batch_id: str) -> int:
    int8_cols = {"target_4h_dir", "target_8h_dir", "target_24h_dir"}

    def cv(val, col):
        if pd.isna(val):
            return None
        if col in int8_cols:
            return int(val)
        return float(val) if not isinstance(val, str) else val

    rows = [tuple(cv(row[c], c) for c in FEATURE_COLS) + (gold_batch_id,) for _, row in df.iterrows()]
    placeholders = ", ".join(["%s"] * (len(FEATURE_COLS) + 1))
    cols = ", ".join(FEATURE_COLS) + ", gold_batch_id"
    update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in FEATURE_COLS if c not in ("ticker", "ts")) + ", computed_at = NOW()"

    sql = f"""
    INSERT INTO gold.ml_features ({cols}) VALUES ({placeholders})
    ON CONFLICT (ticker, ts) DO UPDATE SET {update_set};
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=500)
    conn.commit()
    return len(rows)


def write_feature_audit(conn, audit: dict):
    sql = """
    INSERT INTO gold.feature_audit
      (gold_batch_id, ticker, started_at, finished_at, rows_read, rows_written, rows_dropped, n_features,
       train_rows, val_rows, test_rows, status, error_message)
    VALUES
      (%(gold_batch_id)s, %(ticker)s, %(started_at)s, %(finished_at)s, %(rows_read)s, %(rows_written)s, %(rows_dropped)s,
       %(n_features)s, %(train_rows)s, %(val_rows)s, %(test_rows)s, %(status)s, %(error_message)s);
    """
    with conn.cursor() as cur:
        cur.execute(sql, audit)
    conn.commit()


def main():
    batch_id = str(uuid.uuid4())
    started = datetime.now(timezone.utc)

    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor() as cur:
        cur.execute(GOLD_DDL)
    conn.commit()

    audit = {
        "gold_batch_id": batch_id,
        "ticker": TICKER,
        "started_at": started,
        "finished_at": None,
        "rows_read": 0,
        "rows_written": 0,
        "rows_dropped": 0,
        "n_features": len(FEATURE_COLS) - 6,
        "train_rows": 0,
        "val_rows": 0,
        "test_rows": 0,
        "status": "RUNNING",
        "error_message": None,
    }

    try:
        df = read_silver(conn, TICKER)
        audit["rows_read"] = len(df)

        feat = compute_gold_features(df)
        before = len(feat)
        feat = feat.dropna(subset=["ema_50", "rsi_14", "macd", "bb_pct_b"])
        audit["rows_dropped"] = before - len(feat)

        feat = assign_splits(feat)
        counts = feat["split"].value_counts()
        audit["train_rows"] = int(counts.get("train", 0))
        audit["val_rows"] = int(counts.get("val", 0))
        audit["test_rows"] = int(counts.get("test", 0))

        audit["rows_written"] = write_gold(conn, feat, batch_id)
        audit["status"] = "SUCCESS"
        audit["finished_at"] = datetime.now(timezone.utc)
        write_feature_audit(conn, audit)
        log.info("Gold complete | written=%s", audit["rows_written"])

    except Exception as exc:
        audit["status"] = "FAILED"
        audit["error_message"] = str(exc)
        audit["finished_at"] = datetime.now(timezone.utc)
        write_feature_audit(conn, audit)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
