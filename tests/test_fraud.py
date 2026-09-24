import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_demo_data import make_demo_csv  # noqa: E402
from model import FraudEnsemble  # noqa: E402


def test_train_and_predict_real_scores(tmp_path):
    csv = str(tmp_path / "demo.csv")
    make_demo_csv(csv, n=2000, fraud_rate=0.05)
    df = pd.read_csv(csv)
    model = FraudEnsemble()
    metrics = model.fit(df)
    assert model.trained and metrics["n_samples"] == 2000

    legit = df[df["is_fraud"] == 0].iloc[0].to_dict()
    fraud = df[df["is_fraud"] == 1].iloc[0].to_dict()

    r_legit = model.predict_one(legit)
    r_fraud = model.predict_one(fraud)

    for r in (r_legit, r_fraud):
        assert 0.0 <= r["fraud_probability"] <= 1.0
        assert set(r) >= {"fraud_probability", "is_fraud", "components"}

    # real learned separation: the fraud-pattern row scores higher than a legit row
    assert r_fraud["fraud_probability"] > r_legit["fraud_probability"]

    # scores vary across inputs (not canned)
    probs = {
        round(model.predict_one(df.sample(1).iloc[0].to_dict())["fraud_probability"], 6)
        for _ in range(10)
    }
    assert len(probs) > 1

    # save/load round-trip keeps behavior
    p = str(tmp_path / "m.joblib")
    model.save(p)
    loaded = FraudEnsemble.load(p)
    r2 = loaded.predict_one(fraud)
    assert abs(r2["fraud_probability"] - r_fraud["fraud_probability"]) < 1e-9
