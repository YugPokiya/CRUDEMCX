"""Physics-inspired ANN signal generator for SmartAPI trading workflows.

This module trains a compact ANN on OHLCV-derived features and generates
BUY/SELL/HOLD signals with confidence and risk levels.

Input CSV must contain:
- timestamp, open, high, low, close, volume

Example:
    python ml/physics_ann_signal.py \
      --csv data/raw_ohlcv.csv \
      --symbol CRUDEOIL \
      --epochs 25 \
      --model-out models/physics_ann.keras \
      --signal-out outputs/smartapi_signal.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


@dataclass
class PhysicsArtifacts:
    model: tf.keras.Model
    scaler: StandardScaler
    features: list[str]


def load_ohlcv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {"timestamp", "open", "high", "low", "close", "volume"}
    miss = sorted(needed.difference(df.columns))
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    return df


def build_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1"] = out["close"].pct_change()
    out["ret_3"] = out["close"].pct_change(3)
    out["velocity"] = out["close"].diff()
    out["acceleration"] = out["velocity"].diff()
    out["jerk"] = out["acceleration"].diff()
    out["range_pct"] = (out["high"] - out["low"]) / (out["close"] + 1e-9)
    out["body_pct"] = (out["close"] - out["open"]) / (out["open"] + 1e-9)
    out["kinetic_proxy"] = 0.5 * (out["velocity"] ** 2)
    out["vol_force"] = out["volume"] * out["acceleration"].fillna(0)
    out["energy_flow"] = out["ret_1"].fillna(0) * np.log1p(out["volume"])

    # Future return label for classification
    future_ret = out["close"].shift(-1) / out["close"] - 1
    out["label"] = np.where(future_ret > 0.0025, 2, np.where(future_ret < -0.0025, 0, 1))
    out = out.dropna().reset_index(drop=True)
    return out


def train_physics_ann(df: pd.DataFrame, epochs: int = 25) -> PhysicsArtifacts:
    features = [
        "open", "high", "low", "close", "volume",
        "ret_1", "ret_3", "velocity", "acceleration", "jerk",
        "range_pct", "body_pct", "kinetic_proxy", "vol_force", "energy_flow",
    ]

    x = df[features].values
    y = df["label"].astype(int).values

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    y_cat = tf.keras.utils.to_categorical(y, num_classes=3)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(x_scaled.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        x_scaled,
        y_cat,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    )
    return PhysicsArtifacts(model=model, scaler=scaler, features=features)


def generate_signal(artifacts: PhysicsArtifacts, df: pd.DataFrame, symbol: str) -> dict:
    latest = df.iloc[-1:][artifacts.features].values
    x = artifacts.scaler.transform(latest)
    probs = artifacts.model.predict(x, verbose=0)[0]

    cls = int(np.argmax(probs))
    side = {0: "SELL", 1: "HOLD", 2: "BUY"}[cls]
    confidence = float(probs[cls])

    last_close = float(df.iloc[-1]["close"])
    atr_like = float((df["high"] - df["low"]).tail(14).mean())
    stop = last_close - 1.2 * atr_like if side == "BUY" else last_close + 1.2 * atr_like
    target = last_close + 1.8 * atr_like if side == "BUY" else last_close - 1.8 * atr_like

    return {
        "broker": "smartapi",
        "symbol": symbol,
        "signal": side,
        "confidence": round(confidence, 4),
        "probabilities": {
            "SELL": round(float(probs[0]), 4),
            "HOLD": round(float(probs[1]), 4),
            "BUY": round(float(probs[2]), 4),
        },
        "price": last_close,
        "stop_loss": round(stop, 4),
        "target": round(target, 4),
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Physics ANN signal generator for SmartAPI")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--symbol", default="CRUDEOIL")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--model-out", default="models/physics_ann.keras")
    ap.add_argument("--signal-out", default="outputs/smartapi_signal.json")
    args = ap.parse_args()

    df = load_ohlcv(args.csv)
    feat_df = build_physics_features(df)
    artifacts = train_physics_ann(feat_df, epochs=args.epochs)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    artifacts.model.save(args.model_out)

    signal = generate_signal(artifacts, feat_df, args.symbol)
    Path(args.signal_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.signal_out).write_text(json.dumps(signal, indent=2))

    print(json.dumps(signal, indent=2))
    print(f"Model saved: {args.model_out}")
    print(f"Signal saved: {args.signal_out}")


if __name__ == "__main__":
    main()
