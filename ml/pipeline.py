"""MCX OHLCV medallion pipeline + TensorFlow classifier with backtesting.

Layers:
- Bronze: raw imported OHLCV (as-is with metadata)
- Silver: validated, typed, deduplicated OHLCV with percentage changes
- Gold: model-ready feature table with supervised labels
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
    label_column: str


def _ensure_dirs(base_dir: Path) -> tuple[Path, Path, Path]:
    bronze_dir = base_dir / "bronze"
    silver_dir = base_dir / "silver"
    gold_dir = base_dir / "gold"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    silver_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)
    return bronze_dir, silver_dir, gold_dir


def ingest_to_bronze(raw_csv: str, base_dir: str = "data") -> Path:
    bronze_dir, _, _ = _ensure_dirs(Path(base_dir))
    raw = pd.read_csv(raw_csv)
    raw["ingested_at"] = pd.Timestamp.utcnow()
    raw["source_file"] = Path(raw_csv).name
    bronze_path = bronze_dir / "ohlcv_bronze.csv"
    raw.to_csv(bronze_path, index=False)
    print(f"✓ Bronze ingestion complete: {len(raw)} records")
    return bronze_path


def bronze_to_silver(bronze_path: str, base_dir: str = "data") -> Path:
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

    grouped = df.groupby("commodity", group_keys=False)
    df["pct_open"] = grouped["open"].pct_change(1) * 100
    df["pct_high"] = grouped["high"].pct_change(1) * 100
    df["pct_low"] = grouped["low"].pct_change(1) * 100
    df["pct_close"] = grouped["close"].pct_change(1) * 100
    df["pct_volume"] = grouped["volume"].pct_change(1) * 100
    df["intraday_pct_range"] = ((df["high"] - df["low"]) / (df["close"] + 1e-9) * 100).fillna(0)
    df[["pct_open", "pct_high", "pct_low", "pct_close", "pct_volume"]] = (
        df[["pct_open", "pct_high", "pct_low", "pct_close", "pct_volume"]].fillna(0)
    )

    silver_path = silver_dir / "ohlcv_silver.csv"
    df.to_csv(silver_path, index=False)
    print(f"✓ Silver layer complete: percentage changes calculated for {len(df)} records")
    return silver_path


def _calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return (100 - (100 / (1 + rs))).fillna(50)


def silver_to_gold(silver_path: str, base_dir: str = "data", profit_threshold: float = 0.5) -> Path:
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
    df["volume_z"] = grouped["volume"].transform(lambda s: (s - s.rolling(20).mean()) / (s.rolling(20).std() + 1e-9))
    df["rsi_14"] = grouped["close"].transform(lambda s: _calculate_rsi(s, 14))
    ema_fast = grouped["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema_slow = grouped["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["macd"] = (ema_fast - ema_slow).fillna(0)

    next_day_return = grouped["pct_close"].shift(-1)
    df["label"] = pd.cut(
        next_day_return,
        bins=[-np.inf, -profit_threshold, profit_threshold, np.inf],
        labels=["loss", "neutral", "profit"],
        include_lowest=True,
    )
    label_map = {"loss": 0, "neutral": 1, "profit": 2}
    df["label_encoded"] = df["label"].map(label_map)
    df["next_day_return"] = next_day_return

    df = df.dropna().reset_index(drop=True)

    gold_path = gold_dir / "ohlcv_gold_features.csv"
    df.to_csv(gold_path, index=False)
    print(f"✓ Gold layer complete: {len(df)} records with supervised labels")
    print(f"  Label distribution: {dict(df['label'].value_counts())}")
    return gold_path


def load_gold(gold_csv: str) -> pd.DataFrame:
    df = pd.read_csv(gold_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values(["commodity", "timestamp"]).reset_index(drop=True)


def build_sequences(values: np.ndarray, labels: np.ndarray, lookback: int = 24) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(lookback, len(values)):
        x.append(values[i - lookback : i])
        y.append(labels[i])
    return np.array(x), np.array(y)


def train_lstm_classifier(df: pd.DataFrame, epochs: int = 20, lookback: int = 24) -> TrainArtifacts:
    features = [
        "open", "high", "low", "close", "volume",
        "ret_1", "ret_3", "ret_6",
        "volatility_12", "sma_6", "sma_12", "volume_z",
        "pct_open", "pct_high", "pct_low", "pct_close", "pct_volume",
        "intraday_pct_range", "rsi_14", "macd",
    ]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    labels = df["label_encoded"].astype(int).values
    x, y = build_sequences(scaled, labels, lookback=lookback)

    y_cat = tf.keras.utils.to_categorical(y, num_classes=3)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(x.shape[1], x.shape[2])),
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2),
        tf.keras.layers.LSTM(32, dropout=0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        x,
        y_cat,
        validation_split=0.2,
        epochs=epochs,
        batch_size=16,
        verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    )

    return TrainArtifacts(model=model, scaler=scaler, feature_columns=features, label_column="label_encoded")


def backtest_classifier(artifacts: TrainArtifacts, df: pd.DataFrame, lookback: int = 24) -> dict:
    values = artifacts.scaler.transform(df[artifacts.feature_columns])
    labels = df[artifacts.label_column].astype(int).values
    next_returns = df["next_day_return"].values

    x, y_true = build_sequences(values, labels, lookback=lookback)
    preds = artifacts.model.predict(x, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    acc = float((y_pred == y_true).mean()) if len(y_true) else 0.0

    aligned_returns = next_returns[lookback: lookback + len(y_pred)]
    trade_mask = y_pred != 1  # skip neutral
    signed_returns = np.where(y_pred == 2, aligned_returns, -aligned_returns)
    traded = signed_returns[trade_mask]
    total_return = float(np.nansum(traded)) if len(traded) else 0.0

    equity = np.cumsum(np.nan_to_num(traded, nan=0.0))
    if len(equity):
        peak = np.maximum.accumulate(equity)
        drawdown = equity - peak
        max_drawdown = float(drawdown.min())
    else:
        max_drawdown = 0.0

    return {
        "samples": int(len(y_true)),
        "accuracy": acc,
        "trades": int(trade_mask.sum()),
        "total_return_pct": total_return,
        "max_drawdown_pct": max_drawdown,
    }


def run_medallion_pipeline(raw_csv: str, base_dir: str = "data", profit_threshold: float = 0.5) -> Path:
    bronze = ingest_to_bronze(raw_csv, base_dir)
    silver = bronze_to_silver(str(bronze), base_dir)
    return silver_to_gold(str(silver), base_dir, profit_threshold=profit_threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="MCX OHLCV medallion + classifier + backtest")
    parser.add_argument("--csv", required=True, help="Raw OHLCV CSV path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--profit-threshold", type=float, default=0.5)
    parser.add_argument("--data-dir", default="data", help="Output directory for bronze/silver/gold")
    parser.add_argument("--model-out", default="models/mcx_classifier.keras")
    args = parser.parse_args()

    gold_path = run_medallion_pipeline(args.csv, args.data_dir, args.profit_threshold)
    gold_df = load_gold(str(gold_path))

    artifacts = train_lstm_classifier(gold_df, epochs=args.epochs, lookback=args.lookback)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    artifacts.model.save(args.model_out)

    bt = backtest_classifier(artifacts, gold_df, lookback=args.lookback)

    print(f"Bronze/Silver/Gold built under: {args.data_dir}")
    print(f"Gold dataset: {gold_path}")
    print(f"Model saved to: {args.model_out}")
    print(f"Backtest: samples={bt['samples']} accuracy={bt['accuracy']:.4f} trades={bt['trades']} total_return_pct={bt['total_return_pct']:.4f} max_drawdown_pct={bt['max_drawdown_pct']:.4f}")


if __name__ == "__main__":
    main()
