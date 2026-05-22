"""
CRUDEMCX — FinBERT Sentiment Pipeline
======================================
Produces 4 weekly sentiment features for the Silver → Gold layer:
  1. sent_polarity        — net bullish/bearish direction (-1 to +1)
  2. sent_uncertainty     — epistemic hedging & doubt (0 to 1)
  3. sent_forward_looking — forward-looking language score (0 to 1)
  4. sent_intensity       — confidence magnitude (0 to 1)

Based on: arxiv 2603.11408 (WTI crude oil LightGBM + FinBERT, 2025)
SHAP finding: uncertainty + forward_looking dimensions dominate polarity.

Architecture:
  NewsCollector → TextPreprocessor → FinBERTScorer
  → UncertaintyScorer → ForwardLookingScorer → WeeklyAggregator → CSVWriter

Usage:
  python finbert_pipeline.py --start 2024-01-01 --end 2024-12-31 --out sentiment_features.csv
  python finbert_pipeline.py --demo   # run on synthetic news, no API key needed
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import re
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("crudemcx.sentiment")


# ─────────────────────────────────────────────────────────────
# 1. DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class NewsArticle:
    date: str          # YYYY-MM-DD
    headline: str
    body: str = ""
    source: str = "unknown"
    week: str = ""     # ISO week key: YYYY-Www

    def full_text(self) -> str:
        """Combine headline + first 512 chars of body for scoring."""
        body_snip = (self.body[:400] + "…") if len(self.body) > 400 else self.body
        return f"{self.headline}. {body_snip}".strip()

    def __post_init__(self):
        if self.date and not self.week:
            try:
                d = datetime.strptime(self.date, "%Y-%m-%d")
                self.week = d.strftime("%G-W%V")
            except ValueError:
                pass


@dataclass
class ArticleScores:
    date: str
    week: str
    headline: str
    polarity: float        # -1 = bearish, 0 = neutral, +1 = bullish
    uncertainty: float     # 0–1
    forward_looking: float # 0–1
    intensity: float       # 0–1
    finbert_label: str = ""
    finbert_conf: float = 0.0


@dataclass
class WeeklySentiment:
    week: str
    week_start: str
    n_articles: int
    sent_polarity: float
    sent_uncertainty: float
    sent_forward_looking: float
    sent_intensity: float


# ─────────────────────────────────────────────────────────────
# 2. NEWS COLLECTOR
# ─────────────────────────────────────────────────────────────

MCX_CRUDE_KEYWORDS = [
    "crude oil", "MCX crude", "NCRUDEOIL", "oil price", "OPEC",
    "petroleum", "WTI", "Brent crude", "crude futures", "oil market",
    "energy market", "oil demand", "oil supply", "crude inventory",
    "EIA report", "oil rally", "oil slump", "crude slump", "oil bearish",
    "crude bullish", "OPEC cut", "refinery", "oil outlook",
]


class NewsCollector:
    """
    Fetches crude oil news headlines. Three modes:
      - NewsAPI (requires NEWSAPI_KEY env var)
      - Alphavanage (requires ALPHA_VANTAGE_KEY env var)
      - Demo mode: generates synthetic headlines for testing
    """

    def __init__(self, start: str, end: str):
        self.start = datetime.strptime(start, "%Y-%m-%d")
        self.end   = datetime.strptime(end,   "%Y-%m-%d")

    # ── Public ──────────────────────────────────────────────

    def collect(self, demo: bool = False) -> list[NewsArticle]:
        if demo:
            log.info("Demo mode: generating synthetic headlines")
            return self._synthetic_headlines()

        articles = []
        if key := os.getenv("NEWSAPI_KEY"):
            articles = self._from_newsapi(key)
        elif key := os.getenv("ALPHA_VANTAGE_KEY"):
            articles = self._from_alphavantage(key)
        else:
            log.warning("No API key found. Set NEWSAPI_KEY or ALPHA_VANTAGE_KEY. "
                        "Falling back to demo mode.")
            articles = self._synthetic_headlines()

        log.info(f"Collected {len(articles)} articles ({self.start.date()} → {self.end.date()})")
        return articles

    # ── Sources ─────────────────────────────────────────────

    def _from_newsapi(self, key: str) -> list[NewsArticle]:
        """NewsAPI.org — free tier gives 1 month history."""
        try:
            import requests
        except ImportError:
            log.error("pip install requests")
            return []

        articles = []
        query = " OR ".join(f'"{k}"' for k in MCX_CRUDE_KEYWORDS[:6])
        url = "https://newsapi.org/v2/everything"
        cursor = self.start

        while cursor <= self.end:
            params = {
                "q": query,
                "from": cursor.strftime("%Y-%m-%d"),
                "to":   min(cursor + timedelta(days=6), self.end).strftime("%Y-%m-%d"),
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 100,
                "apiKey": key,
            }
            try:
                r = requests.get(url, params=params, timeout=15)
                data = r.json()
                for a in data.get("articles", []):
                    pub = a.get("publishedAt", "")[:10]
                    articles.append(NewsArticle(
                        date=pub,
                        headline=a.get("title", ""),
                        body=a.get("description", "") or a.get("content", ""),
                        source=a.get("source", {}).get("name", "newsapi"),
                    ))
            except Exception as e:
                log.warning(f"NewsAPI error: {e}")
            cursor += timedelta(days=7)

        return articles

    def _from_alphavantage(self, key: str) -> list[NewsArticle]:
        """Alpha Vantage news sentiment endpoint."""
        try:
            import requests
        except ImportError:
            return []

        url = "https://www.alphavantage.co/query"
        articles = []
        # AV supports time_from/time_to as YYYYMMDDTHHMM
        params = {
            "function": "NEWS_SENTIMENT",
            "topics": "energy_transportation",
            "time_from": self.start.strftime("%Y%m%dT0000"),
            "time_to":   self.end.strftime("%Y%m%dT2359"),
            "limit": 200,
            "apikey": key,
        }
        try:
            r = requests.get(url, params=params, timeout=20)
            data = r.json()
            for item in data.get("feed", []):
                pub = item.get("time_published", "")[:8]
                date = f"{pub[:4]}-{pub[4:6]}-{pub[6:8]}" if len(pub) >= 8 else ""
                # filter for crude oil relevance
                summary = item.get("summary", "")
                headline = item.get("title", "")
                combined = (headline + " " + summary).lower()
                if any(kw.lower() in combined for kw in ["crude", "oil", "opec", "petroleum"]):
                    articles.append(NewsArticle(
                        date=date, headline=headline, body=summary, source="alphavantage"
                    ))
        except Exception as e:
            log.warning(f"AlphaVantage error: {e}")

        return articles

    def _synthetic_headlines(self) -> list[NewsArticle]:
        """Generates plausible synthetic headlines for demo/testing."""
        templates = [
            # Bearish / uncertainty
            ("MCX crude oil tumbles ₹{p} as OPEC output concerns mount", "bearish"),
            ("Crude oil futures fall on demand uncertainty amid global slowdown", "bearish"),
            ("Oil prices may slide further if US inventories disappoint — analysts", "uncertain"),
            ("MCX NCRUDEOIL hits {w}-week low; traders cautious ahead of EIA data", "bearish"),
            ("OPEC+ may reconsider output cuts, clouding crude oil outlook", "uncertain"),
            ("Crude oil outlook uncertain as geopolitical risks mount in Middle East", "uncertain"),
            # Bullish / forward-looking
            ("MCX crude rallies ₹{p} on OPEC supply cut hopes", "bullish"),
            ("Oil prices set to rise as demand recovery expected in Q{q}", "forward"),
            ("Crude futures could test ₹{hi} if EIA shows inventory draw", "forward"),
            ("Analysts forecast WTI at ${wti} by year-end on robust demand", "forward"),
            ("MCX crude likely to outperform on rupee depreciation tailwinds", "forward"),
            ("Oil prices stabilise; OPEC compliance expected to remain above 90%", "bullish"),
            # Neutral
            ("Crude oil trades near flat as markets await Fed rate decision", "neutral"),
            ("MCX NCRUDEOIL settles ₹{p} higher in choppy session", "neutral"),
            ("Oil demand forecast revised; IEA cites mixed signals from China", "neutral"),
        ]
        import random
        random.seed(42)
        articles = []
        cursor = self.start
        while cursor <= self.end:
            # 3–7 articles per week
            n = random.randint(3, 7)
            for _ in range(n):
                tmpl, _ = random.choice(templates)
                headline = tmpl.format(
                    p=random.randint(20, 150),
                    w=random.randint(2, 8),
                    q=random.randint(1, 4),
                    hi=random.randint(6000, 8000),
                    wti=random.randint(70, 95),
                )
                offset = timedelta(days=random.randint(0, 6))
                date = (cursor + offset).strftime("%Y-%m-%d")
                if datetime.strptime(date, "%Y-%m-%d") > self.end:
                    date = cursor.strftime("%Y-%m-%d")
                articles.append(NewsArticle(date=date, headline=headline, source="synthetic"))
            cursor += timedelta(days=7)

        return sorted(articles, key=lambda a: a.date)


# ─────────────────────────────────────────────────────────────
# 3. FINBERT SCORER  (polarity + intensity)
# ─────────────────────────────────────────────────────────────

class FinBERTScorer:
    """
    Wraps ProsusAI/finbert for financial sentiment.
    Labels: positive → bullish (+1), negative → bearish (-1), neutral → 0
    Also extracts confidence as intensity proxy.
    """

    MODEL_ID = "ProsusAI/finbert"

    def __init__(self, batch_size: int = 16, device: Optional[str] = None):
        self.batch_size = batch_size
        self._pipeline = None
        self._device = device

    def _load(self):
        if self._pipeline is not None:
            return
        log.info(f"Loading {self.MODEL_ID} …")
        from transformers import pipeline as hf_pipeline
        import torch

        device = self._device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._pipeline = hf_pipeline(
            "text-classification",
            model=self.MODEL_ID,
            tokenizer=self.MODEL_ID,
            device=0 if device == "cuda" else -1,
            truncation=True,
            max_length=512,
        )
        log.info(f"FinBERT loaded on {device}")

    def score_batch(self, texts: list[str]) -> list[dict]:
        """Returns list of {label, score} dicts."""
        self._load()
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            out = self._pipeline(batch, batch_size=self.batch_size)
            results.extend(out)
        return results

    @staticmethod
    def label_to_polarity(label: str, conf: float) -> float:
        """Convert finbert label → signed polarity."""
        label = label.lower()
        if label == "positive":
            return conf
        elif label == "negative":
            return -conf
        return 0.0


# ─────────────────────────────────────────────────────────────
# 4. UNCERTAINTY SCORER  (lexicon-based)
# ─────────────────────────────────────────────────────────────

# Carefully curated for commodity/energy news context
UNCERTAINTY_LEXICON = {
    # Epistemic hedges
    "may": 0.7, "might": 0.75, "could": 0.7, "uncertain": 0.9, "unclear": 0.85,
    "ambiguous": 0.85, "unpredictable": 0.9, "volatile": 0.8, "risk": 0.65,
    "concern": 0.65, "worry": 0.7, "doubt": 0.8, "possibly": 0.7,
    "perhaps": 0.7, "likely": 0.5, "unlikely": 0.55, "questionable": 0.8,
    "mixed signals": 0.85, "cautious": 0.75, "wait and see": 0.9,
    "if": 0.45, "should": 0.4, "whether": 0.65, "depends": 0.7,
    "remains to be seen": 0.95, "watch": 0.5, "await": 0.6,
    # Volatility language
    "choppy": 0.8, "turbulent": 0.8, "rangebound": 0.6, "sideways": 0.55,
    "pressure": 0.6, "headwinds": 0.7, "tailwinds": 0.5,
}


class UncertaintyScorer:
    """
    Lexicon + pattern based uncertainty scoring.
    Score = weighted average of matched token scores, clipped to [0, 1].
    """

    def __init__(self):
        # Compile regex patterns (multi-word phrases first)
        self._patterns: list[tuple[re.Pattern, float]] = []
        # Sort by length desc so multi-word matches first
        for phrase, weight in sorted(UNCERTAINTY_LEXICON.items(), key=lambda x: -len(x[0])):
            self._patterns.append((re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE), weight))

    def score(self, text: str) -> float:
        if not text:
            return 0.0
        words = len(text.split())
        hits = []
        for pattern, weight in self._patterns:
            matches = pattern.findall(text)
            hits.extend([weight] * len(matches))
        if not hits:
            return 0.0
        # normalize by text length (per 100 words) and clip
        raw = sum(hits) / max(words, 1) * 30
        return float(np.clip(raw, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────
# 5. FORWARD-LOOKING SCORER  (lexicon + tense patterns)
# ─────────────────────────────────────────────────────────────

FORWARD_LEXICON = {
    # Future tense markers
    "will": 0.8, "would": 0.65, "expect": 0.85, "forecast": 0.9,
    "predict": 0.9, "anticipate": 0.85, "project": 0.75, "outlook": 0.8,
    "target": 0.75, "estimate": 0.7, "by year-end": 0.9,
    "next week": 0.85, "next month": 0.85, "next quarter": 0.85,
    "going forward": 0.9, "ahead": 0.7, "upcoming": 0.85,
    "future": 0.8, "in the coming": 0.9, "set to": 0.85,
    "poised to": 0.85, "could reach": 0.8, "may test": 0.8,
    "likely to": 0.75, "expected to": 0.9, "analysts see": 0.8,
    "by q": 0.8, "fy2": 0.85, "h1": 0.6, "h2": 0.6,
    "guidance": 0.7, "roadmap": 0.7,
}


class ForwardLookingScorer:
    """Lexicon + future-tense regex scoring."""

    FUTURE_TENSE = re.compile(
        r"\b(will|shall|going to|set to|poised to|expected to|forecast to)\b",
        re.IGNORECASE
    )
    TEMPORAL_FUTURE = re.compile(
        r"\b(next\s+(week|month|quarter|year)|by\s+(end|year|q[1-4]|fy)|upcoming|"
        r"in\s+the\s+coming|ahead|going\s+forward|forecast)\b",
        re.IGNORECASE
    )

    def __init__(self):
        self._patterns: list[tuple[re.Pattern, float]] = []
        for phrase, weight in sorted(FORWARD_LEXICON.items(), key=lambda x: -len(x[0])):
            self._patterns.append((re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE), weight))

    def score(self, text: str) -> float:
        if not text:
            return 0.0
        words = len(text.split())
        hits = []
        for pattern, weight in self._patterns:
            matches = pattern.findall(text)
            hits.extend([weight] * len(matches))
        # Bonus for future tense patterns
        ft_hits = len(self.FUTURE_TENSE.findall(text)) * 0.85
        tf_hits = len(self.TEMPORAL_FUTURE.findall(text)) * 0.8
        total = sum(hits) + ft_hits + tf_hits
        raw = total / max(words, 1) * 25
        return float(np.clip(raw, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────
# 6. WEEKLY AGGREGATOR
# ─────────────────────────────────────────────────────────────

class WeeklyAggregator:
    """
    Aggregates per-article scores into weekly features.

    Aggregation strategy (matching arxiv 2603.11408):
      - sent_polarity:        EWM-weighted mean (recent articles count more)
      - sent_uncertainty:     90th-percentile (captures peak uncertainty)
      - sent_forward_looking: median (robust to outliers)
      - sent_intensity:       mean confidence of high-polarity articles
    """

    def aggregate(self, scores: list[ArticleScores]) -> list[WeeklySentiment]:
        df = pd.DataFrame([asdict(s) for s in scores])
        if df.empty:
            return []

        results = []
        for week, grp in df.groupby("week"):
            grp = grp.sort_values("date").reset_index(drop=True)
            n = len(grp)

            # EWM weights (most recent article has highest weight)
            weights = np.exp(np.linspace(-1, 0, n))
            weights /= weights.sum()

            polarity        = float(np.dot(grp["polarity"].values, weights))
            uncertainty     = float(np.percentile(grp["uncertainty"].values, 90))
            forward_looking = float(grp["forward_looking"].median())
            # intensity = mean confidence among strongly opinionated articles
            strong = grp[grp["polarity"].abs() > 0.3]["intensity"]
            intensity = float(strong.mean()) if len(strong) > 0 else float(grp["intensity"].mean())

            # Derive week_start from ISO week string
            try:
                week_dt = datetime.strptime(week + "-1", "%G-W%V-%u")
                week_start = week_dt.strftime("%Y-%m-%d")
            except Exception:
                week_start = grp["date"].iloc[0]

            results.append(WeeklySentiment(
                week=week,
                week_start=week_start,
                n_articles=n,
                sent_polarity=round(polarity, 6),
                sent_uncertainty=round(uncertainty, 6),
                sent_forward_looking=round(forward_looking, 6),
                sent_intensity=round(intensity, 6),
            ))

        return sorted(results, key=lambda w: w.week)


# ─────────────────────────────────────────────────────────────
# 7. MAIN PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

class CRUDEMCXSentimentPipeline:
    """
    End-to-end pipeline:
      NewsCollector → FinBERTScorer + UncertaintyScorer + ForwardLookingScorer
      → WeeklyAggregator → DataFrame with 4 weekly features
    """

    def __init__(
        self,
        start: str,
        end: str,
        batch_size: int = 16,
        demo: bool = False,
        device: Optional[str] = None,
    ):
        self.start = start
        self.end = end
        self.demo = demo

        self.collector     = NewsCollector(start, end)
        self.finbert       = FinBERTScorer(batch_size=batch_size, device=device)
        self.uncertainty   = UncertaintyScorer()
        self.forward       = ForwardLookingScorer()
        self.aggregator    = WeeklyAggregator()

    def run(self) -> pd.DataFrame:
        # Step 1: collect
        articles = self.collector.collect(demo=self.demo)
        if not articles:
            log.warning("No articles collected — returning empty DataFrame")
            return pd.DataFrame()

        # Step 2: score with FinBERT (batch)
        texts = [a.full_text() for a in articles]
        log.info(f"Scoring {len(texts)} articles with FinBERT …")
        finbert_out = self.finbert.score_batch(texts)

        # Step 3: build per-article scores
        article_scores: list[ArticleScores] = []
        for article, fb in zip(articles, finbert_out):
            label = fb["label"]
            conf  = float(fb["score"])

            polarity   = FinBERTScorer.label_to_polarity(label, conf)
            uncert     = self.uncertainty.score(article.full_text())
            fwd        = self.forward.score(article.full_text())
            intensity  = conf  # raw FinBERT confidence as intensity

            article_scores.append(ArticleScores(
                date=article.date,
                week=article.week,
                headline=article.headline[:100],
                polarity=round(polarity, 6),
                uncertainty=round(uncert, 6),
                forward_looking=round(fwd, 6),
                intensity=round(intensity, 6),
                finbert_label=label,
                finbert_conf=round(conf, 6),
            ))

        log.info(f"Scored {len(article_scores)} articles")

        # Step 4: weekly aggregation
        weekly = self.aggregator.aggregate(article_scores)
        df = pd.DataFrame([asdict(w) for w in weekly])

        log.info(f"Aggregated into {len(df)} weekly rows")
        return df

    def run_and_save(self, out_path: str) -> pd.DataFrame:
        df = self.run()
        if not df.empty:
            df.to_csv(out_path, index=False)
            log.info(f"Saved → {out_path}")
            self._print_summary(df)
        return df

    @staticmethod
    def _print_summary(df: pd.DataFrame):
        print("\n" + "═" * 60)
        print("  CRUDEMCX SENTIMENT FEATURES — WEEKLY SUMMARY")
        print("═" * 60)
        print(f"  Weeks: {df['week'].iloc[0]} → {df['week'].iloc[-1]}")
        print(f"  Rows:  {len(df)}")
        print()
        features = ["sent_polarity", "sent_uncertainty", "sent_forward_looking", "sent_intensity"]
        print(f"  {'Feature':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print("  " + "-" * 55)
        for f in features:
            print(f"  {f:<25} {df[f].mean():>8.4f} {df[f].std():>8.4f} "
                  f"{df[f].min():>8.4f} {df[f].max():>8.4f}")
        print("═" * 60 + "\n")
        print(df[["week", "week_start", "n_articles"] + features].tail(8).to_string(index=False))
        print()


# ─────────────────────────────────────────────────────────────
# 8. MEDALLION INTEGRATION HELPER
# ─────────────────────────────────────────────────────────────

def merge_into_silver(ohlcv_df: pd.DataFrame, sentiment_path: str) -> pd.DataFrame:
    """
    Merge weekly sentiment features into CRUDEMCX Silver layer OHLCV DataFrame.

    Args:
        ohlcv_df:       Silver-layer DataFrame with a 'date' column (YYYY-MM-DD).
        sentiment_path: Path to the sentiment CSV produced by this pipeline.

    Returns:
        DataFrame with 4 new columns appended, forward-filled for trading days.
    """
    sent = pd.read_csv(sentiment_path, parse_dates=["week_start"])
    sent = sent.rename(columns={"week_start": "date"})
    sent["date"] = pd.to_datetime(sent["date"])

    ohlcv = ohlcv_df.copy()
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    # Left-merge on week boundary, then forward-fill within each week
    merged = pd.merge_asof(
        ohlcv.sort_values("date"),
        sent[["date", "sent_polarity", "sent_uncertainty",
              "sent_forward_looking", "sent_intensity"]].sort_values("date"),
        on="date",
        direction="backward",
    )

    feature_cols = ["sent_polarity", "sent_uncertainty", "sent_forward_looking", "sent_intensity"]
    merged[feature_cols] = merged[feature_cols].fillna(method="ffill")

    return merged


# ─────────────────────────────────────────────────────────────
# 9. CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CRUDEMCX FinBERT Sentiment Pipeline — produces 4 weekly features"
    )
    p.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end",   default="2024-12-31", help="End date YYYY-MM-DD")
    p.add_argument("--out",   default="sentiment_features.csv", help="Output CSV path")
    p.add_argument("--batch-size", type=int, default=16, help="FinBERT batch size")
    p.add_argument("--device", default=None, help="cuda / cpu (auto-detect if omitted)")
    p.add_argument("--demo",   action="store_true",
                   help="Use synthetic headlines (no API key required)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = CRUDEMCXSentimentPipeline(
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        demo=args.demo,
        device=args.device,
    )
    pipeline.run_and_save(args.out)
