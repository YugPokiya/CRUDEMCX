# file: ml_backtest_any_data.py
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf


def load_ohlcv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().sort_values("timestamp").reset_index(drop=True)


def build_features(df: pd.DataFrame, threshold: float = 0.0025) -> pd.DataFrame:
    x = df.copy()
    x["ret_1"] = x["close"].pct_change(1)
    x["ret_3"] = x["close"].pct_change(3)
    x["ret_6"] = x["close"].pct_change(6)
    x["velocity"] = x["close"].diff()
    x["acceleration"] = x["velocity"].diff()
    x["range_pct"] = (x["high"] - x["low"]) / (x["close"] + 1e-9)
    x["body_pct"] = (x["close"] - x["open"]) / (x["open"] + 1e-9)
    x["vol_z"] = (x["volume"] - x["volume"].rolling(20).mean()) / (x["volume"].rolling(20).std() + 1e-9)

    fwd_ret = x["close"].shift(-1) / x["close"] - 1
    # 0=SELL,1=HOLD,2=BUY
    x["label"] = np.where(fwd_ret > threshold, 2, np.where(fwd_ret < -threshold, 0, 1))
    x["next_ret"] = fwd_ret
    return x.dropna().reset_index(drop=True)


def make_model(input_dim: int) -> tf.keras.Model:
    lookback_window = 21
    num_features =34
    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(lookback_window, num_features)),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax"),
])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def backtest(y_pred: np.ndarray, next_ret: np.ndarray):
    # buy => +ret, sell => -ret, hold => 0
    trade_ret = np.where(y_pred == 2, next_ret, np.where(y_pred == 0, -next_ret, 0.0))
    equity = np.cumprod(1.0 + trade_ret)

    trades = (y_pred != 1).sum()
    wins = (trade_ret > 0).sum()
    win_rate = wins / max(1, trades)
    total_return = equity[-1] - 1.0 if len(equity) else 0.0

    peak = np.maximum.accumulate(equity) if len(equity) else np.array([1.0])
    dd = (equity / peak - 1.0) if len(equity) else np.array([0.0])
    max_dd = dd.min()

    # rough annualization for hourly bars: sqrt(24*365)
    sharpe = 0.0
    if trade_ret.std() > 1e-12:
        sharpe = (trade_ret.mean() / trade_ret.std()) * np.sqrt(24 * 365)

    return {
        "trades": int(trades),
        "win_rate": float(win_rate),
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
        "sharpe_approx": float(sharpe),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to OHLCV csv")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.0025, help="Label threshold")
    ap.add_argument("--train-ratio", type=float, default=0.8, help="Time split ratio")
    args = ap.parse_args()

    df = load_ohlcv(args.csv)
    feat = build_features(df, threshold=args.threshold)

    feature_cols = [
        "open", "high", "low", "close", "volume",
        "ret_1", "ret_3", "ret_6", "velocity", "acceleration",
        "range_pct", "body_pct", "vol_z"
    ]

    split_idx = int(len(feat) * args.train_ratio)
    train_df = feat.iloc[:split_idx].copy()
    test_df = feat.iloc[split_idx:].copy()

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_cols].values)
    x_test = scaler.transform(test_df[feature_cols].values)

    y_train = train_df["label"].astype(int).values
    y_test = test_df["label"].astype(int).values
    y_train_cat = tf.keras.utils.to_categorical(y_train)

    model = make_model(x_train.shape[1])
    model.fit(
        x_train, y_train_cat,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=32,
        verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    )

    probs = model.predict(x_test, verbose=0)
    y_pred = probs.argmax(axis=1)

    print("\n=== Classification ===")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["SELL", "HOLD", "BUY"], digits=4))

    stats = backtest(y_pred, test_df["next_ret"].values)
    print("\n=== Backtest ===")
    for k, v in stats.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()