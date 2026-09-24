"""Real fraud-detection ensemble.

Combines two genuinely trained models:
  * IsolationForest (unsupervised)  -> anomaly score from the data itself
  * LogisticRegression (supervised, class-weighted) -> fraud probability, used when
    the training CSV contains the `is_fraud` label column

No simulated inference: every score comes from a model fitted on real input data.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "amount",
    "time_hour",                # 0-23
    "merchant_category",        # integer-encoded category
    "card_present",             # 0 / 1
    "distance_from_home_km",
    "transactions_last_24h",
    "avg_amount_30d",
]
LABEL_COL = "is_fraud"
MODEL_PATH = os.environ.get("FRAUD_MODEL_PATH", "model.joblib")


class FraudEnsemble:
    def __init__(self, contamination=0.05, random_state=42):
        self.scaler = StandardScaler()
        self.iforest = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=random_state
        )
        self.lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        self.uses_supervised = False
        self.trained = False

    # ---------------- training ----------------
    def fit(self, df: pd.DataFrame) -> dict:
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        if len(df) < 20:
            raise ValueError("need at least 20 rows to train")
        X = self.scaler.fit_transform(df[FEATURE_COLS].astype(float).values)
        self.iforest.fit(X)

        metrics = {"n_samples": int(len(df)), "supervised": False}
        if LABEL_COL in df.columns and df[LABEL_COL].nunique() >= 2:
            y = df[LABEL_COL].astype(int).values
            self.lr.fit(X, y)
            self.uses_supervised = True
            p = self.lr.predict(X)
            try:
                auc = float(roc_auc_score(y, self.lr.predict_proba(X)[:, 1]))
            except Exception:
                auc = None
            metrics.update(
                {
                    "supervised": True,
                    "train_precision": float(precision_score(y, p, zero_division=0)),
                    "train_recall": float(recall_score(y, p, zero_division=0)),
                    "train_roc_auc": auc,
                }
            )
        self.trained = True
        return metrics

    # ---------------- inference ----------------
    def _vector(self, row: dict) -> np.ndarray:
        try:
            vals = [float(row[c]) for c in FEATURE_COLS]
        except KeyError as e:
            raise ValueError(f"missing feature: {e}")
        return self.scaler.transform(np.array([vals]))

    def predict_one(self, row: dict) -> dict:
        if not self.trained:
            raise RuntimeError("model not trained")
        x = self._vector(row)
        # higher = more anomalous (real output of the fitted IsolationForest)
        anomaly = float(-self.iforest.decision_function(x)[0])
        anomaly_prob = float(1.0 / (1.0 + np.exp(-anomaly)))
        components = {
            "anomaly_score": anomaly,
            "anomaly_probability": anomaly_prob,
        }
        if self.uses_supervised:
            lr_p = float(self.lr.predict_proba(x)[0][1])
            components["supervised_probability"] = lr_p
            fraud_prob = 0.5 * lr_p + 0.5 * anomaly_prob
        else:
            fraud_prob = anomaly_prob
        return {
            "fraud_probability": fraud_prob,
            "is_fraud": bool(fraud_prob >= 0.5),
            "components": components,
        }

    # ---------------- persistence ----------------
    def save(self, path: str = MODEL_PATH) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = MODEL_PATH) -> "FraudEnsemble":
        return joblib.load(path)
