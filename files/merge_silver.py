"""
CRUDEMCX — Silver Layer Sentiment Merger
=========================================
Merges finbert_pipeline output into your existing Silver-layer OHLCV parquet/CSV.

Usage:
  python merge_silver.py \
    --ohlcv  data/silver/ncrudeoil_silver.csv \
    --sent   sentiment_features.csv \
    --out    data/silver/ncrudeoil_silver_with_sent.csv
"""

import argparse
import pandas as pd
from pathlib import Path


def merge(ohlcv_path: str, sent_path: str, out_path: str):
    # ── Load
    ext = Path(ohlcv_path).suffix.lower()
    if ext == ".parquet":
        ohlcv = pd.read_parquet(ohlcv_path)
    else:
        ohlcv = pd.read_csv(ohlcv_path, parse_dates=["date"])

    sent = pd.read_csv(sent_path, parse_dates=["week_start"])
    sent = sent.rename(columns={"week_start": "date"})

    # ── Sort for merge_asof
    ohlcv = ohlcv.sort_values("date").reset_index(drop=True)
    sent  = sent.sort_values("date").reset_index(drop=True)

    feature_cols = [
        "sent_polarity", "sent_uncertainty",
        "sent_forward_looking", "sent_intensity",
    ]

    merged = pd.merge_asof(
        ohlcv,
        sent[["date"] + feature_cols],
        on="date",
        direction="backward",  # use last known week's sentiment
    )

    # Forward-fill any gaps at start of series
    merged[feature_cols] = merged[feature_cols].ffill()

    # ── Save
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if out_path.endswith(".parquet"):
        merged.to_parquet(out_path, index=False)
    else:
        merged.to_csv(out_path, index=False)

    print(f"Merged {len(merged)} rows → {out_path}")
    print(merged[["date"] + feature_cols].tail(5).to_string(index=False))
    return merged


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ohlcv", required=True)
    p.add_argument("--sent",  required=True)
    p.add_argument("--out",   required=True)
    args = p.parse_args()
    merge(args.ohlcv, args.sent, args.out)
