"""
CRUDEMCX | Bronze Layer (Medallion Architecture)

Purpose:
- Land raw OHLCV as received with full provenance.
- Add quality flags and quarantine markers without silently dropping data.
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

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

RAW_FILE = Path(os.getenv("RAW_FILE", Path(__file__).resolve().parents[2] / "data" / "raw" / "ncrudeoil_hourly_raw.csv"))
SOURCE_SYS = os.getenv("SOURCE_SYS", "YFINANCE_CL_EQ_F")
TICKER = os.getenv("TICKER", "NCRUDEOIL")

PRICE_MIN = float(os.getenv("PRICE_MIN", "1000"))
PRICE_MAX = float(os.getenv("PRICE_MAX", "20000"))

BRONZE_DDL = """
CREATE SCHEMA IF NOT EXISTS bronze;

CREATE TABLE IF NOT EXISTS bronze.ohlcv_raw (
    row_id          BIGSERIAL PRIMARY KEY,
    ticker          VARCHAR(30)  NOT NULL,
    ts              TIMESTAMPTZ  NOT NULL,
    open            NUMERIC(14,4),
    high            NUMERIC(14,4),
    low             NUMERIC(14,4),
    close           NUMERIC(14,4),
    volume          BIGINT,
    source_system   VARCHAR(60)  NOT NULL,
    batch_id        UUID         NOT NULL,
    ingested_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    quality_flags   TEXT[]       NOT NULL DEFAULT '{}',
    is_duplicate    BOOLEAN      NOT NULL DEFAULT FALSE,
    is_quarantined  BOOLEAN      NOT NULL DEFAULT FALSE,
    UNIQUE (ticker, ts)
);

CREATE INDEX IF NOT EXISTS idx_bronze_ticker_ts ON bronze.ohlcv_raw (ticker, ts DESC);
CREATE INDEX IF NOT EXISTS idx_bronze_batch ON bronze.ohlcv_raw (batch_id);
CREATE INDEX IF NOT EXISTS idx_bronze_quarantine ON bronze.ohlcv_raw (is_quarantined) WHERE is_quarantined = TRUE;

CREATE TABLE IF NOT EXISTS bronze.ingest_audit (
    audit_id         BIGSERIAL PRIMARY KEY,
    batch_id         UUID         NOT NULL,
    ticker           VARCHAR(30),
    source_system    VARCHAR(60),
    started_at       TIMESTAMPTZ  NOT NULL,
    finished_at      TIMESTAMPTZ,
    rows_in_file     INT,
    rows_deduped     INT,
    rows_inserted    INT,
    rows_updated     INT,
    rows_flagged     INT,
    rows_quarantined INT,
    status           VARCHAR(20)  DEFAULT 'RUNNING',
    error_message    TEXT
);
"""


def flag_quality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["quality_flags"] = [[] for _ in range(len(df))]
    df["is_quarantined"] = False

    for col in ["open", "high", "low", "close", "volume"]:
        mask = df[col].isna()
        df.loc[mask, "quality_flags"] = df.loc[mask, "quality_flags"].apply(lambda f: f + [f"NULL_{col.upper()}"])
        if col != "volume":
            df.loc[mask, "is_quarantined"] = True

    for col in ["open", "high", "low", "close"]:
        mask = df[col].notna() & ((df[col] < PRICE_MIN) | (df[col] > PRICE_MAX))
        df.loc[mask, "quality_flags"] = df.loc[mask, "quality_flags"].apply(lambda f: f + [f"{col.upper()}_OUT_OF_RANGE"])
        df.loc[mask, "is_quarantined"] = True

    valid_mask = df[["open", "high", "low", "close"]].notna().all(axis=1)

    high_below_low = valid_mask & (df["high"] < df["low"])
    df.loc[high_below_low, "quality_flags"] = df.loc[high_below_low, "quality_flags"].apply(lambda f: f + ["HIGH_BELOW_LOW"])
    df.loc[high_below_low, "is_quarantined"] = True

    close_outside = valid_mask & ((df["close"] > df["high"]) | (df["close"] < df["low"]))
    df.loc[close_outside, "quality_flags"] = df.loc[close_outside, "quality_flags"].apply(lambda f: f + ["CLOSE_OUTSIDE_HL"])
    df.loc[close_outside, "is_quarantined"] = True

    zero_volume = df["volume"].notna() & (df["volume"] <= 0)
    df.loc[zero_volume, "quality_flags"] = df.loc[zero_volume, "quality_flags"].apply(lambda f: f + ["ZERO_VOLUME"])

    df_sorted = df.sort_values("ts")
    spike_idx = df_sorted.index[df_sorted["close"].pct_change().abs() > 0.15]
    df.loc[spike_idx, "quality_flags"] = df.loc[spike_idx, "quality_flags"].apply(lambda f: f + ["PRICE_SPIKE_GT15PCT"])
    df.loc[spike_idx, "is_quarantined"] = True

    df["quality_flags"] = df["quality_flags"].apply(lambda lst: "{" + ",".join(lst) + "}" if lst else "{}")
    return df


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.sort_values("ts")
    df["is_duplicate"] = df.duplicated(subset=["ticker", "ts"], keep="first")
    n_duped = int(df["is_duplicate"].sum())
    df = df[~df["is_duplicate"]].copy()
    df["is_duplicate"] = False
    log.info("Deduplication: %s rows in -> %s unique rows (%s removed)", before, len(df), n_duped)
    return df, n_duped


def _safe(row: pd.Series, col: str):
    v = row.get(col, np.nan)
    return None if pd.isna(v) else float(v)


def upsert_bronze(conn, df: pd.DataFrame, batch_id: str, source_system: str) -> int:
    now = datetime.now(timezone.utc)
    rows = [
        (
            row["ticker"], row["ts"], _safe(row, "open"), _safe(row, "high"), _safe(row, "low"), _safe(row, "close"),
            None if pd.isna(row.get("volume", np.nan)) else int(row["volume"]),
            source_system, batch_id, now, row["quality_flags"], bool(row["is_duplicate"]), bool(row["is_quarantined"]),
        )
        for _, row in df.iterrows()
    ]

    sql = """
    INSERT INTO bronze.ohlcv_raw
      (ticker, ts, open, high, low, close, volume, source_system, batch_id, ingested_at, quality_flags, is_duplicate, is_quarantined)
    VALUES %s
    ON CONFLICT (ticker, ts) DO NOTHING;
    """

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, page_size=500)
    conn.commit()
    return len(rows)


def write_audit(conn, audit: dict):
    sql = """
    INSERT INTO bronze.ingest_audit
      (batch_id, ticker, source_system, started_at, finished_at, rows_in_file, rows_deduped, rows_inserted, rows_updated,
       rows_flagged, rows_quarantined, status, error_message)
    VALUES
      (%(batch_id)s, %(ticker)s, %(source_system)s, %(started_at)s, %(finished_at)s, %(rows_in_file)s, %(rows_deduped)s,
       %(rows_inserted)s, %(rows_updated)s, %(rows_flagged)s, %(rows_quarantined)s, %(status)s, %(error_message)s);
    """
    with conn.cursor() as cur:
        cur.execute(sql, audit)
    conn.commit()


def main():
    batch_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)
    log.info("Bronze ingest start | batch_id=%s", batch_id)

    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor() as cur:
        cur.execute(BRONZE_DDL)
    conn.commit()

    audit = {
        "batch_id": batch_id,
        "ticker": TICKER,
        "source_system": SOURCE_SYS,
        "started_at": started_at,
        "finished_at": None,
        "rows_in_file": 0,
        "rows_deduped": 0,
        "rows_inserted": 0,
        "rows_updated": 0,
        "rows_flagged": 0,
        "rows_quarantined": 0,
        "status": "RUNNING",
        "error_message": None,
    }

    try:
        df = pd.read_csv(RAW_FILE, parse_dates=["ts"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        if "ticker" not in df.columns:
            df["ticker"] = TICKER

        audit["rows_in_file"] = len(df)

        df, n_duped = deduplicate(df)
        audit["rows_deduped"] = n_duped

        df = flag_quality(df)
        audit["rows_flagged"] = int((df["quality_flags"] != "{}").sum())
        audit["rows_quarantined"] = int(df["is_quarantined"].sum())

        audit["rows_inserted"] = upsert_bronze(conn, df, batch_id, SOURCE_SYS)

        audit["status"] = "SUCCESS"
        audit["finished_at"] = datetime.now(timezone.utc)
        write_audit(conn, audit)
        log.info("Bronze complete | inserted=%s", audit["rows_inserted"])

    except Exception as exc:
        audit["status"] = "FAILED"
        audit["error_message"] = str(exc)
        audit["finished_at"] = datetime.now(timezone.utc)
        write_audit(conn, audit)
        log.error("Bronze failed: %s", exc)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
