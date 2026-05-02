"""
Score an observation sequence against trained HMMs and return a prediction.

This module is the runtime classification engine. It loads every <gesture>.pkl
from models/ at startup, then exposes:

  - GestureClassifier(models_dir).predict(symbols) -> (label or None, score, margin)
  - GestureClassifier(models_dir).score_all(symbols) -> dict[label -> log_prob]

A confidence margin is enforced: if the gap between the best and second-best
log-likelihood is below `min_margin`, predict() returns (None, ...) instead of
the top guess. This kills false positives on ambiguous motion.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np


class GestureClassifier:
    def __init__(self, models_dir: Path | str = "models", min_margin: float = 1.0):
        self.models_dir = Path(models_dir)
        self.min_margin = float(min_margin)
        self.models: dict[str, object] = {}
        self._load()

    def _load(self) -> None:
        if not self.models_dir.exists():
            raise FileNotFoundError(f"models dir not found: {self.models_dir}")
        for pkl in sorted(self.models_dir.glob("*.pkl")):
            with pkl.open("rb") as f:
                payload = pickle.load(f)
            gesture = payload.get("gesture", pkl.stem)
            self.models[gesture] = payload["model"]
        if not self.models:
            raise RuntimeError(f"no models found under {self.models_dir} - run train.py first.")

    def gestures(self) -> list[str]:
        return sorted(self.models.keys())

    def score_all(self, symbols: np.ndarray) -> dict[str, float]:
        """Return log-likelihood of `symbols` under every trained HMM."""
        if len(symbols) == 0:
            return {g: float("-inf") for g in self.models}
        X = np.asarray(symbols, dtype=np.int64).reshape(-1, 1)
        return {g: float(m.score(X)) for g, m in self.models.items()}

    def predict(self, symbols: np.ndarray) -> tuple[Optional[str], float, float]:
        """
        Predict the gesture label for a symbol sequence.

        Returns
        -------
        (label, best_log_prob, margin)
            label : str or None  - None if margin < min_margin
            best_log_prob : float
            margin : float       - best minus second-best log-prob
        """
        scores = self.score_all(symbols)
        if not scores:
            return None, float("-inf"), 0.0

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_label, best_lp = ranked[0]
        second_lp = ranked[1][1] if len(ranked) > 1 else float("-inf")
        margin = best_lp - second_lp

        if margin < self.min_margin:
            return None, best_lp, margin
        return best_label, best_lp, margin
