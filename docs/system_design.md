# MCX Trading Platform - System Design

Objective: Build an intelligent trading assistant for MCX commodities with OHLCV + hourly news ingestion, RNN/LSTM modeling, risk-tagged levels, and a dashboard reasoning tab.

## Data Pipeline (Bronze → Silver → Gold)
1. **Bronze**: raw imported OHLCV records with ingestion metadata (`source_file`, `ingested_at`).
2. **Silver**: cleaned/typed/deduplicated OHLCV records with strict schema and quality checks.
3. **Gold**: model-ready features (`ret_1`, `ret_3`, `ret_6`, `volatility_12`, `sma_6`, `sma_12`, `volume_z`) for ML training/inference.

Architecture: Rust ingestion/API, Python TensorFlow ML, PostgreSQL persistence, React frontend.

Required software: Rust/Cargo, PostgreSQL, Node.js/npm, Python 3.11+, TensorFlow, pandas, numpy, scikit-learn.
