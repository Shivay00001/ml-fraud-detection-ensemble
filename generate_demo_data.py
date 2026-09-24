"""DEMO DATA GENERATOR — SYNTHETIC DATA, NOT REAL TRANSACTIONS.

Produces a labeled transaction CSV with a known fraud pattern so the model can be
tested end-to-end. NEVER present this file as real data.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)


def make_demo_csv(path: str, n: int = 5000, fraud_rate: float = 0.04) -> str:
    legit = int(n * (1 - fraud_rate))
    fraud = n - legit

    def block(k, fraud_flag):
        df = pd.DataFrame(
            {
                "amount": RNG.lognormal(3.2, 0.9, k),
                "time_hour": RNG.integers(0, 24, k),
                "merchant_category": RNG.integers(0, 12, k),
                "card_present": RNG.integers(0, 2, k),
                "distance_from_home_km": RNG.exponential(8, k),
                "transactions_last_24h": RNG.poisson(3, k),
                "avg_amount_30d": RNG.lognormal(3.0, 0.8, k),
                "is_fraud": fraud_flag,
            }
        )
        if fraud_flag:
            # fraud pattern: big amounts, odd hours, far from home, card not present
            df["amount"] = RNG.lognormal(5.2, 0.8, k)
            df["time_hour"] = RNG.choice([1, 2, 3, 4, 23], k)
            df["distance_from_home_km"] = RNG.exponential(120, k)
            df["card_present"] = 0
        return df

    df = pd.concat([block(legit, 0), block(fraud, 1)], ignore_index=True)
    df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    out = make_demo_csv("demo_transactions.csv")
    print(f"wrote SYNTHETIC DEMO data -> {out} (NOT real transactions)")
