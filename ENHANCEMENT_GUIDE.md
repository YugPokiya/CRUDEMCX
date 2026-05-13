# MCX Pipeline Enhancement Guide

## What changed
- Silver layer now computes percentage features (`pct_open`, `pct_high`, `pct_low`, `pct_close`, `pct_volume`, `intraday_pct_range`).
- Gold layer now creates supervised labels (`loss`, `neutral`, `profit`) using next-day `pct_close` and `--profit-threshold`.
- Training switched to multi-class classification with softmax output.
- Added backtest summary metrics (accuracy, trades, cumulative return %, max drawdown %).

## Run
```bash
python ml/pipeline.py --csv data/raw_ohlcv.csv --epochs 20 --lookback 24 --profit-threshold 0.5
```

## Backtest output
The script now prints:
- `samples`
- `accuracy`
- `trades`
- `total_return_pct`
- `max_drawdown_pct`
