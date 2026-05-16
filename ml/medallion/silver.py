"""CRUDEMCX | Silver Layer (Medallion Architecture)."""

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
SPIKE_WINDOW = int(os.getenv("SPIKE_WINDOW", "24"))

SILVER_DDL = """
CREATE SCHEMA IF NOT EXISTS silver;

CREATE TABLE IF NOT EXISTS silver.ohlcv_clean (
    row_id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(30) NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    open NUMERIC(14,4) NOT NULL,
    high NUMERIC(14,4) NOT NULL,
    low NUMERIC(14,4) NOT NULL,
    close NUMERIC(14,4) NOT NULL,
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
"""


class RepairLog:
    def __init__(self, silver_batch_id: str):
        self.batch_id = silver_batch_id
        self.records = []

    def add(self, ticker, ts, field, issue, strategy, original, repaired):
        self.records.append({
            "silver_batch_id": self.batch_id,
            "ticker": ticker,
            "ts": ts,
            "field": field,
            "issue": issue,
            "strategy": strategy,
            "original_value": float(original) if original is not None and not pd.isna(original) else None,
            "repaired_value": float(repaired) if repaired is not None and not pd.isna(repaired) else None,
        })

    def write(self, conn):
        if not self.records:
            return
        sql = """
        INSERT INTO silver.repair_log
          (silver_batch_id, ticker, ts, field, issue, strategy, original_value, repaired_value)
        VALUES %s;
        """
        rows = [(
            r["silver_batch_id"], r["ticker"], r["ts"], r["field"], r["issue"], r["strategy"], r["original_value"], r["repaired_value"],
        ) for r in self.records]
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, rows, page_size=500)
        conn.commit()


def read_bronze(conn, ticker: str) -> pd.DataFrame:
    sql = """
    SELECT row_id, ticker, ts, open, high, low, close, volume, quality_flags, is_quarantined,
           batch_id AS bronze_batch_id, source_system
    FROM bronze.ohlcv_raw
    WHERE ticker = %s
    ORDER BY ts ASC;
    """
    df = pd.read_sql(sql, conn, params=(ticker,), parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def repair_bad_prices(df: pd.DataFrame, rlog: RepairLog) -> pd.DataFrame:
    rolling_med = df["close"].rolling(SPIKE_WINDOW, min_periods=1).median()
    for col in ["open", "high", "low", "close"]:
        oor = df["quality_flags"].apply(lambda f: f"{col.upper()}_OUT_OF_RANGE" in str(f))
        spike = df["quality_flags"].apply(lambda f: "PRICE_SPIKE_GT15PCT" in str(f))
        for idx in df[oor | spike].index:
            orig = df.at[idx, col]
            rep = round(float(rolling_med.iloc[idx]), 4)
            df.at[idx, col] = rep
            rlog.add(df.at[idx, "ticker"], df.at[idx, "ts"], col, "OUTLIER_OR_SPIKE", "ROLLING_MEDIAN_REPLACE", orig, rep)
    return df


def repair_null_prices(df: pd.DataFrame, rlog: RepairLog) -> pd.DataFrame:
    for col in ["open", "high", "low", "close"]:
        null_mask = df[col].isna()
        if null_mask.any():
            df[col] = df[col].ffill()
            for idx in df[null_mask].index:
                rlog.add(df.at[idx, "ticker"], df.at[idx, "ts"], col, f"NULL_{col.upper()}", "FORWARD_FILL", None, df.at[idx, col])
    return df


def repair_null_volume(df: pd.DataFrame, rlog: RepairLog) -> pd.DataFrame:
    null_mask = df["volume"].isna()
    if not null_mask.any():
        return df
    rolling_vol = df["volume"].rolling(SPIKE_WINDOW, min_periods=1).median()
    df.loc[null_mask, "volume"] = rolling_vol[null_mask].round(0).astype("Int64")
    for idx in df[null_mask].index:
        rlog.add(df.at[idx, "ticker"], df.at[idx, "ts"], "volume", "NULL_VOLUME", "ROLLING_MEDIAN_FILL", None, df.at[idx, "volume"])
    return df


def enforce_ohlc_logic(df: pd.DataFrame, rlog: RepairLog) -> pd.DataFrame:
    high_fix = df["high"] < df[["open", "close"]].max(axis=1)
    for idx in df[high_fix].index:
        orig = df.at[idx, "high"]
        df.at[idx, "high"] = df.loc[idx, ["open", "close"]].max()
        rlog.add(df.at[idx, "ticker"], df.at[idx, "ts"], "high", "POST_REPAIR_HL_VIOLATION", "SET_TO_MAX_OC", orig, df.at[idx, "high"])

    low_fix = df["low"] > df[["open", "close"]].min(axis=1)
    for idx in df[low_fix].index:
        orig = df.at[idx, "low"]
        df.at[idx, "low"] = df.loc[idx, ["open", "close"]].min()
        rlog.add(df.at[idx, "ticker"], df.at[idx, "ts"], "low", "POST_REPAIR_HL_VIOLATION", "SET_TO_MIN_OC", orig, df.at[idx, "low"])
    return df


def generate_expected_hours(df: pd.DataFrame) -> pd.DatetimeIndex:
    start = df["ts"].min().normalize()
    end = df["ts"].max().normalize() + pd.Timedelta(days=1)
    all_hours = pd.date_range(start=start, end=end, freq="h", tz="UTC")

    def is_mcx_trading(ts):
        t = ts.tz_convert("Asia/Kolkata")
        if t.dayofweek == 6:
            return False
        if t.dayofweek == 5:
            return 9 <= t.hour <= 13
        return 9 <= t.hour <= 23

    return all_hours[[is_mcx_trading(ts) for ts in all_hours]]


def fill_time_gaps(df: pd.DataFrame, ticker: str, rlog: RepairLog):
    expected = generate_expected_hours(df)
    existing = set(df["ts"].tolist())
    missing = [ts for ts in expected if ts not in existing]
    if not missing:
        df["was_gap_filled"] = False
        return df, 0

    df_sorted = df.sort_values("ts").reset_index(drop=True)
    rows = []
    for ts in missing:
        prev = df_sorted[df_sorted["ts"] < ts]
        prev_close = df_sorted["close"].iloc[0] if prev.empty else prev["close"].iloc[-1]
        rows.append({
            "ticker": ticker, "ts": ts, "open": prev_close, "high": prev_close, "low": prev_close, "close": prev_close,
            "volume": 0, "quality_flags": "{}", "is_quarantined": False, "was_gap_filled": True,
            "bronze_batch_id": None, "source_system": "GAP_FILL_SYNTHETIC",
        })
        rlog.add(ticker, ts, "all", "MISSING_TRADING_HOUR", "SYNTHETIC_BAR_FFILL", None, prev_close)

    df["was_gap_filled"] = False
    out = pd.concat([df, pd.DataFrame(rows)], ignore_index=True).sort_values("ts").reset_index(drop=True)
    return out, len(missing)


def validate_silver(df: pd.DataFrame):
    issues = []
    for col in ["open", "high", "low", "close"]:
        n = int(df[col].isna().sum())
        if n:
            issues.append(f"FAIL: {n} null in {col}")
    if (df["high"] < df["low"]).any():
        issues.append("FAIL: high < low rows exist")
    if (df["close"] > df["high"]).any():
        issues.append("FAIL: close > high rows exist")
    if (df["close"] < df["low"]).any():
        issues.append("FAIL: close < low rows exist")
    if df["ts"].duplicated().any():
        issues.append("FAIL: duplicate timestamps")
    return issues


def write_silver(conn, df: pd.DataFrame, silver_batch_id: str) -> int:
    rows = [(
        row["ticker"], row["ts"], float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]),
        int(row["volume"]) if not pd.isna(row["volume"]) else 0, row.get("bronze_batch_id"), row.get("source_system"),
        bool(row.get("was_repaired", False)), bool(row.get("was_gap_filled", False)), row.get("repair_summary"), silver_batch_id,
    ) for _, row in df.iterrows()]

    sql = """
    INSERT INTO silver.ohlcv_clean
      (ticker, ts, open, high, low, close, volume, bronze_batch_id, source_system, was_repaired, was_gap_filled, repair_summary, silver_batch_id)
    VALUES %s
    ON CONFLICT (ticker, ts) DO UPDATE SET
      open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close, volume=EXCLUDED.volume,
      was_repaired=EXCLUDED.was_repaired, was_gap_filled=EXCLUDED.was_gap_filled, repair_summary=EXCLUDED.repair_summary,
      silver_batch_id=EXCLUDED.silver_batch_id, loaded_at=NOW();
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, page_size=500)
    conn.commit()
    return len(rows)


def write_transform_audit(conn, audit: dict):
    sql = """
    INSERT INTO silver.transform_audit
      (silver_batch_id, bronze_batch_id, ticker, started_at, finished_at, rows_read_bronze, rows_quarantined,
       rows_repaired, rows_gap_filled, rows_written, status, error_message)
    VALUES
      (%(silver_batch_id)s, %(bronze_batch_id)s, %(ticker)s, %(started_at)s, %(finished_at)s, %(rows_read_bronze)s,
       %(rows_quarantined)s, %(rows_repaired)s, %(rows_gap_filled)s, %(rows_written)s, %(status)s, %(error_message)s);
    """
    with conn.cursor() as cur:
        cur.execute(sql, audit)
    conn.commit()


def main():
    silver_batch_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)

    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor() as cur:
        cur.execute(SILVER_DDL)
    conn.commit()

    rlog = RepairLog(silver_batch_id)
    audit = {
        "silver_batch_id": silver_batch_id,
        "bronze_batch_id": None,
        "ticker": TICKER,
        "started_at": started_at,
        "finished_at": None,
        "rows_read_bronze": 0,
        "rows_quarantined": 0,
        "rows_repaired": 0,
        "rows_gap_filled": 0,
        "rows_written": 0,
        "status": "RUNNING",
        "error_message": None,
    }

    try:
        df = read_bronze(conn, TICKER)
        audit["rows_read_bronze"] = len(df)
        audit["rows_quarantined"] = int(df["is_quarantined"].sum())
        if "bronze_batch_id" in df.columns and not df["bronze_batch_id"].isna().all():
            audit["bronze_batch_id"] = str(df["bronze_batch_id"].dropna().iloc[0])

        df = repair_bad_prices(df, rlog)
        df = repair_null_prices(df, rlog)
        df = repair_null_volume(df, rlog)
        df = enforce_ohlc_logic(df, rlog)

        repaired_ts = set(r["ts"] for r in rlog.records)
        df["was_repaired"] = df["ts"].isin(repaired_ts)
        df["repair_summary"] = df.apply(
            lambda row: "; ".join(
                f"{r['field']}:{r['strategy']}" for r in rlog.records if r["ts"] == row["ts"]
            ) or None,
            axis=1,
        )
        audit["rows_repaired"] = int(df["was_repaired"].sum())

        df, gaps = fill_time_gaps(df, TICKER, rlog)
        audit["rows_gap_filled"] = gaps

        issues = validate_silver(df)
        if any(i.startswith("FAIL") for i in issues):
            raise ValueError(f"Silver validation failed: {issues}")

        audit["rows_written"] = write_silver(conn, df, silver_batch_id)
        rlog.write(conn)

        audit["status"] = "SUCCESS"
        audit["finished_at"] = datetime.now(timezone.utc)
        write_transform_audit(conn, audit)
        log.info("Silver complete | written=%s", audit["rows_written"])

    except Exception as exc:
        audit["status"] = "FAILED"
        audit["error_message"] = str(exc)
        audit["finished_at"] = datetime.now(timezone.utc)
        write_transform_audit(conn, audit)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
