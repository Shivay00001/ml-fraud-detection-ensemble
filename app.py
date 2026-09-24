"""FastAPI service: real fraud-detection inference from a trained ensemble."""

import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import FEATURE_COLS, LABEL_COL, MODEL_PATH, FraudEnsemble

app = FastAPI(title="Fraud Detection Ensemble", version="1.0.0")
_MODEL: FraudEnsemble | None = None


def get_model() -> FraudEnsemble:
    global _MODEL
    if _MODEL is None:
        if os.path.exists(MODEL_PATH):
            _MODEL = FraudEnsemble.load(MODEL_PATH)
        else:
            raise HTTPException(
                status_code=503, detail="model not trained yet — POST /train first"
            )
    return _MODEL


class TrainRequest(BaseModel):
    csv_path: str


class PredictRequest(BaseModel):
    transaction: dict


@app.get("/health")
def health():
    return {"status": "ok", "model_trained": os.path.exists(MODEL_PATH)}


@app.get("/schema")
def schema():
    return {"features": FEATURE_COLS, "label": LABEL_COL}


@app.post("/train")
def train(req: TrainRequest):
    if not os.path.exists(req.csv_path):
        raise HTTPException(status_code=404, detail=f"CSV not found: {req.csv_path}")
    global _MODEL
    df = pd.read_csv(req.csv_path)
    model = FraudEnsemble()
    train_metrics = model.fit(df)
    model.save(MODEL_PATH)
    _MODEL = model
    return {"status": "trained", "model_path": MODEL_PATH, "metrics": train_metrics}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return get_model().predict_one(req.transaction)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
