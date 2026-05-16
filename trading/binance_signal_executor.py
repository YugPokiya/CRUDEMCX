"""Binance order executor using RSI + Fisher strategy and Physics ANN confirmation.

- Pulls OHLCV from Binance
- Computes TradingView-equivalent RSI+Fisher signals
- Optionally confirms with Physics ANN classifier probabilities
- Places MARKET BUY/SELL orders on Binance spot (or test mode)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import ccxt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


@dataclass
class PhysicsArtifacts:
    model: tf.keras.Model
    scaler: StandardScaler
    features: list[str]


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def fisher_transform(df: pd.DataFrame, length: int = 9) -> tuple[pd.Series, pd.Series]:
    hl2 = (df["high"] + df["low"]) / 2.0
    high_ = hl2.rolling(length).max()
    low_ = hl2.rolling(length).min()
    div = (high_ - low_).replace(0, 1e-3)

    val = pd.Series(0.0, index=df.index)
    fisher = pd.Series(0.0, index=df.index)
    trigger = pd.Series(0.0, index=df.index)

    for i in range(1, len(df)):
        raw_val = 0.66 * (((hl2.iloc[i] - low_.iloc[i]) / div.iloc[i]) - 0.5) + 0.67 * val.iloc[i - 1]
        v = max(min(raw_val, 0.999), -0.999)
        val.iloc[i] = v
        fisher.iloc[i] = 0.5 * np.log((1 + v) / (1 - v)) + 0.5 * fisher.iloc[i - 1]
        trigger.iloc[i] = fisher.iloc[i - 1]
    return fisher, trigger


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def physics_features(df: pd.DataFrame) -> pd.DataFrame:
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
    return out.dropna().reset_index(drop=True)


def train_physics_ann(df: pd.DataFrame, epochs: int = 10) -> PhysicsArtifacts:
    features = [
        "open", "high", "low", "close", "volume",
        "ret_1", "ret_3", "velocity", "acceleration", "jerk",
        "range_pct", "body_pct", "kinetic_proxy", "vol_force", "energy_flow",
    ]
    future_ret = df["close"].shift(-1) / df["close"] - 1
    labels = np.where(future_ret > 0.0025, 2, np.where(future_ret < -0.0025, 0, 1))
    work = df.iloc[:-1].copy()
    y = labels[:-1]

    scaler = StandardScaler()
    x = scaler.fit_transform(work[features])
    y_cat = tf.keras.utils.to_categorical(y, num_classes=3)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(x.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x, y_cat, epochs=epochs, batch_size=32, verbose=0)
    return PhysicsArtifacts(model=model, scaler=scaler, features=features)


def ann_side(artifacts: PhysicsArtifacts, feat_df: pd.DataFrame) -> tuple[str, float]:
    x = artifacts.scaler.transform(feat_df.iloc[-1:][artifacts.features])
    probs = artifacts.model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {0: "SELL", 1: "HOLD", 2: "BUY"}[idx], float(probs[idx])


def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def place_order(exchange: ccxt.Exchange, symbol: str, side: str, quote_size: float, test_mode: bool):
    ticker = exchange.fetch_ticker(symbol)
    last = ticker["last"]
    amount = quote_size / last
    if test_mode:
        return {"test_mode": True, "symbol": symbol, "side": side, "amount": amount, "price": last}
    return exchange.create_order(symbol=symbol, type="market", side=side.lower(), amount=amount)


def main():
    p = argparse.ArgumentParser(description="Binance RSI+Fisher + Physics ANN order executor")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--rsi-length", type=int, default=14)
    p.add_argument("--fisher-length", type=int, default=9)
    p.add_argument("--quote-size", type=float, default=50.0)
    p.add_argument("--ann-epochs", type=int, default=10)
    p.add_argument("--ann-min-confidence", type=float, default=0.52)
    p.add_argument("--test-mode", action="store_true")
    args = p.parse_args()

    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    if not args.test_mode and (not api_key or not api_secret):
        raise ValueError("Set BINANCE_API_KEY and BINANCE_API_SECRET for live orders")

    exchange = ccxt.binance({"apiKey": api_key, "secret": api_secret, "enableRateLimit": True})
    df = fetch_ohlcv(exchange, args.symbol, args.timeframe, args.limit)

    rsi_v = rsi(df["close"], args.rsi_length)
    fisher, trigger = fisher_transform(df, args.fisher_length)
    long_sig = crossover(fisher, trigger) & (rsi_v < 55)
    short_sig = crossunder(fisher, trigger) & (rsi_v > 45)

    feat = physics_features(df)
    ann = train_physics_ann(feat, epochs=args.ann_epochs)
    ann_signal, ann_conf = ann_side(ann, feat)

    action = "HOLD"
    if long_sig.iloc[-1] and ann_signal == "BUY" and ann_conf >= args.ann_min_confidence:
        action = "BUY"
    elif short_sig.iloc[-1] and ann_signal == "SELL" and ann_conf >= args.ann_min_confidence:
        action = "SELL"

    print({
        "symbol": args.symbol,
        "rsi": float(rsi_v.iloc[-1]),
        "fisher": float(fisher.iloc[-1]),
        "trigger": float(trigger.iloc[-1]),
        "ann_signal": ann_signal,
        "ann_confidence": round(ann_conf, 4),
        "final_action": action,
    })

    if action in {"BUY", "SELL"}:
        result = place_order(exchange, args.symbol, action, args.quote_size, args.test_mode)
        print({"order_result": result})


if __name__ == "__main__":
    main()
