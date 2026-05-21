# ─────────────────────────────────────────
# FINBERT PIPELINE SETUP
# ─────────────────────────────────────────

import torch
import requests
import pandas as pd
import numpy as np
import json
import os
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from pycoingecko import CoinGeckoAPI
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# ENVIRONMENT VARIABLES (.env file)
# ─────────────────────────────────────────
# ETHERSCAN_API_KEY=8UPVA71VBEU71YB8INPUNNTNT1FHHTIFEW
# BSCSCAN_API_KEY= --
# SOLSCAN_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjcmVhdGVkQXQiOjE3NzkyNTk4MzYzMjksImVtYWlsIjoieXVnZHBva2l5YUBnbWFpbC5jb20iLCJhY3Rpb24iOiJ0b2tlbi1hcGkiLCJhcGlWZXJzaW9uIjoidjIiLCJpYXQiOjE3NzkyNTk4MzZ9.N0k2m9m_t9Zf0sDawcB_jknQcBaKaRex7o1hb2tLhy4
# GLASSNODE_API_KEY=your_key (optional)
# CRYPTOPANIC_API_KEY= --
# LUNARCRUSH_API_KEY=your_key (optional)

class FinBERTAnalyzer:
    """
    Loads ProsusAI/finbert — fine-tuned on financial
    news for positive/negative/neutral classification.
    Adapted here for crypto-specific text analysis.
    """

    def __init__(self):
        self.model_name = "ProsusAI/finbert"
        self.device     = 0 if torch.cuda.is_available() else -1

        print(f"[FinBERT] Loading model: {self.model_name}")
        print(f"[FinBERT] Device: {'GPU' if self.device == 0 else 'CPU'}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )

        # Sentiment pipeline
        self.pipe = pipeline(
            task            = "text-classification",
            model           = self.model,
            tokenizer       = self.tokenizer,
            device          = self.device,
            return_all_scores = True,    # get all 3 class scores
            truncation      = True,
            max_length      = 512
        )

        # Label mapping from FinBERT output
        self.label_map = {
            "positive": +1.0,
            "negative": -1.0,
            "neutral":   0.0
        }

        print("[FinBERT] Model loaded successfully")


    def analyze_text(self, text: str) -> dict:
        """
        Analyze a single text string.
        Returns label, scores, and numeric sentiment value.
        """
        if not text or len(text.strip()) < 5:
            return {
                "label":    "neutral",
                "score":     0.0,
                "positive":  0.333,
                "negative":  0.333,
                "neutral":   0.333,
                "confidence": 0.333
            }

        try:
            results    = self.pipe(text)[0]
            scores     = {r["label"].lower(): r["score"] for r in results}
            top_label  = max(scores, key=scores.get)
            numeric    = (
                scores.get("positive", 0) -
                scores.get("negative", 0)
            )

            return {
                "label":      top_label,
                "score":      round(numeric, 4),       # -1 to +1
                "positive":   round(scores.get("positive", 0), 4),
                "negative":   round(scores.get("negative", 0), 4),
                "neutral":    round(scores.get("neutral",  0), 4),
                "confidence": round(scores.get(top_label, 0), 4)
            }

        except Exception as e:
            return {
                "label": "neutral", "score": 0.0,
                "error": str(e)
            }


    def analyze_batch(
        self,
        texts: list[str],
        batch_size: int = 16
    ) -> list[dict]:
        """
        Batch analyze multiple texts efficiently.
        Processes in chunks to avoid OOM on CPU.
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch   = texts[i:i+batch_size]
            outputs = self.pipe(batch)
            for text, output in zip(batch, outputs):
                scores    = {r["label"].lower(): r["score"] for r in output}
                top_label = max(scores, key=scores.get)
                numeric   = (
                    scores.get("positive", 0) -
                    scores.get("negative", 0)
                )
                results.append({
                    "text":       text[:100] + "...",
                    "label":      top_label,
                    "score":      round(numeric, 4),
                    "confidence": round(scores.get(top_label, 0), 4)
                })
        return results


    def aggregate_sentiment(
        self,
        analyses: list[dict],
        weights:  list[float] = None
    ) -> dict:
        """
        Aggregate multiple sentiment scores into
        a single weighted composite sentiment.
        """
        if not analyses:
            return {"composite_score": 0.0, "label": "neutral"}

        scores = [a.get("score", 0.0) for a in analyses]

        if weights and len(weights) == len(scores):
            composite = np.average(scores, weights=weights)
        else:
            composite = np.mean(scores)

        positive_count = sum(1 for a in analyses if a.get("label") == "positive")
        negative_count = sum(1 for a in analyses if a.get("label") == "negative")
        neutral_count  = sum(1 for a in analyses if a.get("label") == "neutral")

        if composite > 0.15:
            label = "BULLISH"
        elif composite < -0.15:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        return {
            "composite_score": round(float(composite), 4),
            "label":           label,
            "positive_count":  positive_count,
            "negative_count":  negative_count,
            "neutral_count":   neutral_count,
            "total_analyzed":  len(analyses),
            "bull_bear_ratio": round(
                positive_count / max(negative_count, 1), 2
            )
        }