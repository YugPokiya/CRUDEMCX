# CRUDEMCX Binance Execution Backend

This repository now focuses on **backend trading execution**:
- Rust backend API (`backend/`)
- Binance strategy executor (`trading/binance_signal_executor.py`)
- PostgreSQL schema (`db/`)

## Backend quick start
1. `psql mcx_trade -f db/schema.sql`
2. `cd backend && DATABASE_URL=postgres://<user>:<pass>@<host>:5432/mcx_trade DB_ACQUIRE_TIMEOUT_SECS=8 BIND_ADDR=0.0.0.0:8080 BIND_RETRY_PORTS=20 cargo run`
3. `curl http://localhost:8080/health`
4. `curl http://localhost:8080/health/db`

## Binance buy/sell order execution
Install deps:
```bash
pip install -r trading/requirements.txt
```

Run strategy in test mode (no real order placement):
```bash
python trading/binance_signal_executor.py \
  --symbol BTC/USDT \
  --timeframe 1h \
  --quote-size 50 \
  --test-mode
```

Run live mode (real Binance market orders):
```bash
export BINANCE_API_KEY=<key>
export BINANCE_API_SECRET=<secret>
python trading/binance_signal_executor.py \
  --symbol BTC/USDT \
  --timeframe 1h \
  --quote-size 50
```

Strategy logic:
- Primary signal: RSI + Fisher Transform crossover/crossunder
- Confirmation: Physics-inspired ANN classifier
- Final action: BUY/SELL only when both strategy + ANN agree and confidence threshold passes
