import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def main(out_path: str, n: int, seed: int):
    rng = np.random.default_rng(seed)

    def rchoice(options, p=None, size=n):
        return rng.choice(options, size=size, replace=True, p=p)

    customerID = [f"C{100000+i}" for i in range(n)]
    gender = rchoice(["Male", "Female"])
    SeniorCitizen = rng.integers(0, 2, size=n)
    Partner = rchoice(["Yes", "No"], p=[0.45, 0.55])
    Dependents = rchoice(["Yes", "No"], p=[0.35, 0.65])
    tenure = rng.integers(0, 73, size=n)
    PhoneService = rchoice(["Yes", "No"], p=[0.9, 0.1])
    MultipleLines = rchoice(["Yes", "No", "No phone service"], p=[0.45, 0.45, 0.10])
    InternetService = rchoice(["DSL", "Fiber optic", "No"], p=[0.4, 0.5, 0.1])
    OnlineSecurity = rchoice(["Yes", "No", "No internet service"], p=[0.35, 0.55, 0.10])
    OnlineBackup = rchoice(["Yes", "No", "No internet service"], p=[0.35, 0.55, 0.10])
    DeviceProtection = rchoice(["Yes", "No", "No internet service"], p=[0.35, 0.55, 0.10])
    TechSupport = rchoice(["Yes", "No", "No internet service"], p=[0.35, 0.55, 0.10])
    StreamingTV = rchoice(["Yes", "No", "No internet service"], p=[0.45, 0.45, 0.10])
    StreamingMovies = rchoice(["Yes", "No", "No internet service"], p=[0.45, 0.45, 0.10])
    Contract = rchoice(["Month-to-month", "One year", "Two year"], p=[0.6, 0.25, 0.15])
    PaperlessBilling = rchoice(["Yes", "No"], p=[0.7, 0.3])
    PaymentMethod = rchoice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        p=[0.4, 0.2, 0.2, 0.2]
    )

    base_charge = (
        np.where(InternetService == "No", 20, 35) +
        np.where(InternetService == "Fiber optic", 20, 0) +
        np.where(PhoneService == "Yes", 5, 0) +
        np.where(StreamingTV == "Yes", 7, 0) +
        np.where(StreamingMovies == "Yes", 7, 0) +
        rng.normal(0, 5, size=n)
    )
    MonthlyCharges = np.clip(base_charge, 15, None).round(2)
    TotalCharges = (MonthlyCharges * tenure + rng.normal(0, 35, size=n)).clip(0).round(2)

    logit = (
        -1.5
        + 0.9 * (Contract == "Month-to-month").astype(float)
        - 0.6 * (Contract == "Two year").astype(float)
        + 0.7 * (PaymentMethod == "Electronic check").astype(float)
        + 0.6 * (InternetService == "Fiber optic").astype(float)
        + 0.012 * (MonthlyCharges - MonthlyCharges.mean())
        - 0.03 * (tenure / 12.0)
        + 0.25 * (PaperlessBilling == "Yes").astype(float)
        + 0.2 * (SeniorCitizen == 1).astype(float)
        - 0.15 * (Partner == "Yes").astype(float)
        - 0.1 * (Dependents == "Yes").astype(float)
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    Churn = (rng.uniform(0, 1, size=n) < prob_churn).astype(int)
    ChurnLabel = np.where(Churn == 1, "Yes", "No")

    df = pd.DataFrame({
        "customerID": customerID,
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "Churn": ChurnLabel
    })

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"✅ Saved dataset to {out.resolve()} | shape={df.shape} | churn_rate={round((df['Churn']=='Yes').mean(), 4)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/churn_dataset.csv")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    main(args.out, args.n, args.seed)