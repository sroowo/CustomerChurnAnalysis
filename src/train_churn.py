# src/train_churn.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support
)

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL = [
    "gender","Partner","Dependents","PhoneService","MultipleLines","InternetService",
    "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
    "StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"
]
TARGET = "Churn"

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def build_prep():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )

def build_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=200),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }

def evaluate(name, pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

    print("\n==============================")
    print(name)
    print("ROC-AUC:", round(auc, 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    return {"Model": name, "ROC_AUC": auc, "Precision": prec, "Recall": rec, "F1": f1}

def main(data_path, out_dir, test_size, seed, pos_label="Yes"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    y = (df[TARGET] == pos_label).astype(int)
    X = df[NUMERIC + CATEGORICAL]

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=test_size, random_state=seed, stratify=y
    )

    preprocessor = build_prep()
    models = build_models()
    metrics = []

    pipelines = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", model)])
        pipe.fit(X_train, y_train)
        metrics.append(evaluate(name, pipe, X_test, y_test))
        pipelines[name] = pipe

    metrics_df = pd.DataFrame(metrics).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    print("\nModel comparison:\n", metrics_df)

    best_name = metrics_df.iloc[0]["Model"]
    best_pipe = pipelines[best_name]
    joblib.dump(best_pipe, out / "churn_best_model.pkl")
    print(f"\nSaved best model: {best_name} -> { (out / 'churn_best_model.pkl').resolve() }")

    # Save predictions (Tableau-ready)
    test_probs = best_pipe.predict_proba(X_test)[:, 1]
    test_pred = np.where(test_probs >= 0.5, "Yes", "No")

    pred_df = df_test[["customerID"] + NUMERIC + CATEGORICAL + [TARGET]].copy()
    pred_df["churn_probability"] = test_probs
    pred_df["churn_pred"] = test_pred
    pred_df.to_csv(out / "churn_predictions.csv", index=False)
    print(f"Saved predictions CSV -> { (out / 'churn_predictions.csv').resolve() }")

    # Feature importance / coefficients (top 20)
    try:
        ohe = best_pipe.named_steps["prep"].named_transformers_["cat"]
        feat_names = NUMERIC + list(ohe.get_feature_names_out(CATEGORICAL))
        clf = best_pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
        else:
            importances = None

        if importances is not None:
            top = (
                pd.DataFrame({"feature": feat_names, "importance": importances})
                .sort_values("importance", ascending=False)
                .head(20)
            )
            top.to_csv(out / "churn_top_features.csv", index=False)
            print(f"Saved top features -> { (out / 'churn_top_features.csv').resolve() }")
    except Exception as e:
        print("Skipped feature importance export:", e)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/churn_dataset.csv", help="Path to CSV with churn data")
    ap.add_argument("--out", default="artifacts", help="Output directory for model & predictions")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.data, args.out, args.test_size, args.seed)