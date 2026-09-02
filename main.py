from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import shap

app = FastAPI(title="Fraud Detection API")

# Load the trained model and SHAP explainer once, when the server starts
model = joblib.load('fraud_model.pkl')
explainer = shap.TreeExplainer(model)

# Defines exactly what data a request must include
class Transaction(BaseModel):
    tx_amount: float
    customer_avg_amount: float
    amount_vs_avg_ratio: float
    terminal_risk_count: float
    tx_hour: int

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is running. Go to /docs to test it."}

@app.post("/score")
def score_transaction(tx: Transaction):
    features = np.array([[tx.tx_amount, tx.customer_avg_amount,
                           tx.amount_vs_avg_ratio, tx.terminal_risk_count, tx.tx_hour]])
    probability = model.predict_proba(features)[0][1]
    prediction = "fraud" if probability > 0.5 else "not fraud"
    return {"fraud_probability": round(float(probability), 4), "prediction": prediction}

@app.post("/explain")
def explain_transaction(tx: Transaction):
    features = np.array([[tx.tx_amount, tx.customer_avg_amount,
                           tx.amount_vs_avg_ratio, tx.terminal_risk_count, tx.tx_hour]])
    shap_values = explainer.shap_values(features)

    if isinstance(shap_values, list):
        shap_values_fraud = shap_values[1][0]
    else:
        shap_values_fraud = shap_values[0]

    feature_names = ['tx_amount', 'customer_avg_amount', 'amount_vs_avg_ratio', 'terminal_risk_count', 'tx_hour']
    explanation = {name: round(float(val), 4) for name, val in zip(feature_names, shap_values_fraud)}

    return {"shap_contributions": explanation}