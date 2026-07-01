#!/usr/bin/env python3
"""
Generates pre-computed example predictions for the client-side demo
(demo/index.html). Runs the locally trained model on a handful of
representative customer profiles and saves the results as JSON.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXAMPLES = [
    {
        "label": "New customer, fiber optic, month-to-month",
        "data": {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 190.0,
        },
    },
    {
        "label": "Long-term customer, two-year contract, DSL",
        "data": {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
            "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
            "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
            "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 55.0, "TotalCharges": 3300.0,
        },
    },
    {
        "label": "Senior citizen, no internet, one-year contract",
        "data": {
            "gender": "Female", "SeniorCitizen": 1, "Partner": "Yes", "Dependents": "No",
            "tenure": 24, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "No", "OnlineSecurity": "No internet service",
            "OnlineBackup": "No internet service", "DeviceProtection": "No internet service",
            "TechSupport": "No internet service", "StreamingTV": "No internet service",
            "StreamingMovies": "No internet service", "Contract": "One year",
            "PaperlessBilling": "No", "PaymentMethod": "Mailed check",
            "MonthlyCharges": 20.0, "TotalCharges": 480.0,
        },
    },
    {
        "label": "High-risk: new, high charges, electronic check",
        "data": {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 1, "PhoneService": "Yes", "MultipleLines": "Yes",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 105.0, "TotalCharges": 105.0,
        },
    },
    {
        "label": "Stable family plan, moderate tenure",
        "data": {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
            "tenure": 36, "PhoneService": "Yes", "MultipleLines": "Yes",
            "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "No",
            "DeviceProtection": "Yes", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "Yes",
            "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 70.0, "TotalCharges": 2520.0,
        },
    },
]


def main():
    from src.feature_engineering import FeatureEngineer

    models_dir = ROOT / "models"
    model = joblib.load(models_dir / "model.pkl")
    fe = FeatureEngineer.load(str(models_dir / "feature_engineer.pkl"))

    results = []
    for ex in EXAMPLES:
        df = pd.DataFrame([ex["data"]])
        df_t = fe.transform(df)
        pred = int(model.predict(df_t)[0])
        proba = float(model.predict_proba(df_t)[0][1])
        results.append(
            {
                "label": ex["label"],
                "input": ex["data"],
                "churn_prediction": pred,
                "probability": round(proba, 4),
                "message": "Customer will CHURN" if pred == 1 else "Customer will STAY",
            }
        )

    out_path = ROOT / "demo" / "examples.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} examples to {out_path}")
    for r in results:
        print(f"  - {r['label']}: {r['message']} ({r['probability']*100:.1f}%)")


if __name__ == "__main__":
    main()
