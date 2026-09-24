# ml-fraud-detection-ensemble

**Real** fraud-detection ensemble (rebuilt 2026-09-24 — the previous version was a
simulated-inference stub and has been deleted). Trains two genuine models on your
data and serves real scores:

- **IsolationForest** (unsupervised) → anomaly score learned from the data
- **LogisticRegression** (supervised, class-weighted) → fraud probability, used when
  the CSV has an `is_fraud` label column

`/predict` returns a real `fraud_probability` in [0,1] computed by the trained
models — never canned values.

## Expected CSV columns

| column | meaning |
|---|---|
| `amount` | transaction amount |
| `time_hour` | hour of day, 0–23 |
| `merchant_category` | integer-encoded merchant category |
| `card_present` | 0 / 1 |
| `distance_from_home_km` | km from cardholder home |
| `transactions_last_24h` | count |
| `avg_amount_30d` | cardholder's 30-day average amount |
| `is_fraud` | **optional** label, 0 / 1 (enables supervised branch) |

## Quickstart

```bash
pip install -r requirements.txt

# 1. make SYNTHETIC demo data (for testing only — NOT real transactions)
python generate_demo_data.py

# 2. train
curl -X POST localhost:8001/train -H 'Content-Type: application/json' \
  -d '{"csv_path": "demo_transactions.csv"}'

# 3. predict
curl -X POST localhost:8001/predict -H 'Content-Type: application/json' \
  -d '{"transaction": {"amount": 2500, "time_hour": 3, "merchant_category": 5,
       "card_present": 0, "distance_from_home_km": 300,
       "transactions_last_24h": 9, "avg_amount_30d": 45}}'

# tests
pytest -q
```

Serve: `uvicorn app:app --port 8001`

## Honest limits

- Demo CSV is **synthetic**; the model is only as good as the real data you train it on.
- No holdout evaluation in `/train` — use `train_test_split` yourself for real metrics.
- This is a screening tool, not a compliance-grade fraud system.
