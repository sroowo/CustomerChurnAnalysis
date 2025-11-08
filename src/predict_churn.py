# src/predict_churn.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL = [
    "gender","Partner","Dependents","PhoneService","MultipleLines","InternetService",
    "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
    "StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"
]
TARGET = "Churn"  # optional in incoming file

def main(model_path, data_path, out_path):
    pipe = joblib.load(model_path)
    df = pd.read_csv(data_path)

    need_cols = NUMERIC + CATEGORICAL
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")

    probs = pipe.predict_proba(df[need_cols])[:, 1]
    preds = np.where(probs >= 0.5, "Yes", "No")

    out_df = df.copy()
    out_df["churn_probability"] = probs
    out_df["churn_pred"] = preds

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Saved scored file -> {out.resolve()} | rows={len(out_df)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/churn_best_model.pkl")
    ap.add_argument("--data", required=True, help="Path to CSV to score")
    ap.add_argument("--out", default="artifacts/scored.csv")
    args = ap.parse_args()
    main(args.model, args.data, args.out)