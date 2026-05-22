# CRUDEMCX — FinBERT Sentiment Pipeline

Produces **4 weekly sentiment features** for the CRUDEMCX Silver → Gold layer,
based on MCX crude oil news headlines scored by FinBERT + lexicon analysis.

```
sent_polarity        — net bullish/bearish direction  (-1 to +1)
sent_uncertainty     — epistemic hedging & doubt       ( 0 to 1 )
sent_forward_looking — forward-looking language score  ( 0 to 1 )
sent_intensity       — confidence magnitude            ( 0 to 1 )
```

Academic basis: arxiv 2603.11408 — SHAP analysis on WTI crude LightGBM shows
`sent_uncertainty` and `sent_forward_looking` dominate `sent_polarity` in
feature importance for commodity return prediction.

---

## Quick start

```bash
pip install -r requirements_sentiment.txt

# Demo mode (no API key, synthetic headlines)
python finbert_pipeline.py --demo --start 2024-01-01 --end 2024-12-31

# With NewsAPI (free key at newsapi.org)
export NEWSAPI_KEY=your_key_here
python finbert_pipeline.py --start 2024-01-01 --end 2024-12-31 --out sentiment_features.csv

# With Alpha Vantage (free key, better energy coverage)
export ALPHA_VANTAGE_KEY=your_key_here
python finbert_pipeline.py --start 2024-01-01 --end 2024-12-31 --out sentiment_features.csv
```

## Merge into Silver layer

```bash
python merge_silver.py \
  --ohlcv data/silver/ncrudeoil_silver.csv \
  --sent  sentiment_features.csv \
  --out   data/silver/ncrudeoil_silver_with_sent.csv
```

Or programmatically in your Medallion pipeline:

```python
from finbert_pipeline import merge_into_silver

silver_df = pd.read_parquet("data/silver/ncrudeoil.parquet")
silver_with_sent = merge_into_silver(silver_df, "sentiment_features.csv")
```

---

## Architecture

```
NewsCollector
  ├── NewsAPI          (NEWSAPI_KEY)
  ├── Alpha Vantage    (ALPHA_VANTAGE_KEY)
  └── Demo/Synthetic   (--demo flag)
      ↓
TextPreprocessor       (headline + body[:400])
      ↓
FinBERTScorer          (ProsusAI/finbert → polarity + intensity)
      ↓
UncertaintyScorer      (lexicon-based, 40-token uncertainty vocabulary)
      ↓
ForwardLookingScorer   (lexicon + future-tense regex)
      ↓
WeeklyAggregator
  ├── sent_polarity        → EWM-weighted mean (recent articles weight more)
  ├── sent_uncertainty     → 90th-percentile (captures peak uncertainty)
  ├── sent_forward_looking → median (robust to outliers)
  └── sent_intensity       → mean confidence of high-polarity articles
      ↓
CSV output  →  merge_silver.py  →  Silver layer OHLCV
```

## Output schema

| Column               | Type    | Range    | Description                        |
|----------------------|---------|----------|------------------------------------|
| week                 | str     | ISO week | e.g. 2024-W11                      |
| week_start           | date    | —        | Monday of that ISO week            |
| n_articles           | int     | ≥ 1      | Articles scored that week          |
| sent_polarity        | float   | −1 to +1 | Net bullish/bearish                |
| sent_uncertainty     | float   | 0 to 1   | Epistemic hedging level            |
| sent_forward_looking | float   | 0 to 1   | Future-oriented language           |
| sent_intensity       | float   | 0 to 1   | Confidence magnitude               |

## LightGBM integration (Gold layer)

```python
SENTIMENT_FEATURES = [
    "sent_polarity",
    "sent_uncertainty",
    "sent_forward_looking",
    "sent_intensity",
]

# Add to your existing feature list in gold_layer.py
ALL_FEATURES = EXISTING_FEATURES + SENTIMENT_FEATURES

# Mark as is — no categorical encoding needed (all continuous)
model = lgb.train(
    params,
    lgb.Dataset(X_train[ALL_FEATURES], label=y_train),
    ...
)
```

## Notes

- FinBERT model: `ProsusAI/finbert` (~438MB, downloads once to ~/.cache/huggingface)
- GPU optional but recommended for large date ranges (batch_size=32 on GPU is ~10x faster)
- uncertainty and forward_looking are computed on CPU regardless of device setting
- For MCX-specific news, prefer Alpha Vantage (`energy_transportation` topic) over NewsAPI
- The 90th-percentile aggregation for uncertainty means one very uncertain headline per week
  is enough to elevate the weekly score — matching how a single "OPEC surprise" article
  dominates a trader's weekly risk perception
