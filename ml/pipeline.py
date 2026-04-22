"""MCX OHLCV medallion pipeline + TensorFlow model training.

Layers:
- Bronze: raw imported OHLCV (as-is with metadata)
- Silver: validated, typed, deduplicated OHLCV
- Gold: model-ready feature table
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


@dataclass
class TrainArtifacts:
    model: tf.keras.Model
    scaler: MinMaxScaler
    feature_columns: list[str]


def _ensure_dirs(base_dir: Path) -> tuple[Path, Path, Path]:
    bronze_dir = base_dir / "bronze"
    silver_dir = base_dir / "silver"
    gold_dir = base_dir / "gold"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    silver_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)
    return bronze_dir, silver_dir, gold_dir


def ingest_to_bronze(raw_csv: str, base_dir: str = "data") -> Path:
    """Import raw OHLCV into Bronze layer without transformation."""
    bronze_dir, _, _ = _ensure_dirs(Path(base_dir))
    raw = pd.read_csv(raw_csv)
    raw["ingested_at"] = pd.Timestamp.utcnow()
    raw["source_file"] = Path(raw_csv).name

    bronze_path = bronze_dir / "ohlcv_bronze.csv"
    raw.to_csv(bronze_path, index=False)
    return bronze_path


def bronze_to_silver(bronze_path: str, base_dir: str = "data") -> Path:
    """Validate and standardize Bronze OHLCV into Silver layer."""
    _, silver_dir, _ = _ensure_dirs(Path(base_dir))
    df = pd.read_csv(bronze_path)

    required = {"timestamp", "open", "high", "low", "close", "volume", "commodity"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in bronze data: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", *numeric_cols, "commodity"])
    df = df[df["volume"] >= 0]
    df = df.drop_duplicates(subset=["commodity", "timestamp"], keep="last")
    df = df.sort_values(["commodity", "timestamp"]).reset_index(drop=True)

    silver_path = silver_dir / "ohlcv_silver.csv"
    df.to_csv(silver_path, index=False)
    return silver_path


def silver_to_gold(silver_path: str, base_dir: str = "data") -> Path:
    """Create model-ready features from Silver OHLCV into Gold layer."""
    _, _, gold_dir = _ensure_dirs(Path(base_dir))
    df = pd.read_csv(silver_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = df.sort_values(["commodity", "timestamp"]).reset_index(drop=True)
    grouped = df.groupby("commodity", group_keys=False)

    df["ret_1"] = grouped["close"].pct_change(1)
    df["ret_3"] = grouped["close"].pct_change(3)
    df["ret_6"] = grouped["close"].pct_change(6)
    df["volatility_12"] = grouped["close"].rolling(12).std().reset_index(level=0, drop=True)
    df["sma_6"] = grouped["close"].rolling(6).mean().reset_index(level=0, drop=True)
    df["sma_12"] = grouped["close"].rolling(12).mean().reset_index(level=0, drop=True)
    df["volume_z"] = (
        grouped["volume"].transform(lambda s: (s - s.rolling(20).mean()) / (s.rolling(20).std() + 1e-9))
    )

    df = df.dropna().reset_index(drop=True)

    gold_path = gold_dir / "ohlcv_gold_features.csv"
    df.to_csv(gold_path, index=False)
    return gold_path


def load_gold(gold_csv: str) -> pd.DataFrame:
    df = pd.read_csv(gold_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values(["commodity", "timestamp"]).reset_index(drop=True)


def build_sequences(values: np.ndarray, lookback: int = 24):
    x, y = [], []
    for i in range(lookback, len(values)):
        x.append(values[i - lookback : i])
        y.append(values[i, 3])
    return np.array(x), np.array(y)


def train_lstm(df: pd.DataFrame, epochs: int = 15) -> TrainArtifacts:
    features = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ret_1",
        "ret_3",
        "ret_6",
        "volatility_12",
        "sma_6",
        "sma_12",
        "volume_z",
    ]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    x, y = build_sequences(scaled, lookback=24)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(x.shape[1], x.shape[2])),
            tf.keras.layers.SimpleRNN(32, return_sequences=True),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(x, y, validation_split=0.2, epochs=epochs, batch_size=16, verbose=1)

    return TrainArtifacts(model=model, scaler=scaler, feature_columns=features)


def suggest_levels(artifacts: TrainArtifacts, latest_window: pd.DataFrame) -> dict:
    x = artifacts.scaler.transform(latest_window[artifacts.feature_columns])
    x = np.expand_dims(x, axis=0)
    pred_close_scaled = artifacts.model.predict(x, verbose=0)[0][0]

    baseline = latest_window.iloc[-1]["close"]
    projected = baseline * (1 + (pred_close_scaled - 0.5) / 10)
    risk = "Safety" if projected > baseline else "Danger"

    return {
        "entry": float(baseline),
        "take_profit": float(projected * 1.01),
        "stop_loss": float(projected * 0.99),
        "risk_tag": risk,
    }


def run_medallion_pipeline(raw_csv: str, base_dir: str = "data") -> Path:
    bronze = ingest_to_bronze(raw_csv, base_dir)
    silver = bronze_to_silver(str(bronze), base_dir)
    gold = silver_to_gold(str(silver), base_dir)
    return gold


def main() -> None:
    parser = argparse.ArgumentParser(description="MCX OHLCV medallion + LSTM trainer")
    parser.add_argument("--csv", required=True, help="Raw OHLCV CSV path")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--data-dir", default="data", help="Output directory for bronze/silver/gold")
    parser.add_argument("--model-out", default="models/mcx_lstm.keras")
    args = parser.parse_args()

    gold_path = run_medallion_pipeline(args.csv, args.data_dir)
    gold_df = load_gold(str(gold_path))

    artifacts = train_lstm(gold_df, epochs=args.epochs)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    artifacts.model.save(args.model_out)

    print(f"Bronze/Silver/Gold built under: {args.data_dir}")
    print(f"Gold dataset: {gold_path}")
    print(f"Model saved to: {args.model_out}")


if __name__ == "__main__":
    main()
