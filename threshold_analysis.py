import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score

# Load model and data, same cleaning as before
model = joblib.load('fraud_model.pkl')
df = pd.read_csv('features.csv')

df = df.dropna(subset=['customer_avg_amount', 'amount_vs_avg_ratio', 'terminal_risk_count'])
df['amount_vs_avg_ratio'] = df['amount_vs_avg_ratio'].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['amount_vs_avg_ratio'])
df['amount_vs_avg_ratio'] = df['amount_vs_avg_ratio'].clip(upper=20)

df = df.sort_values('tx_datetime')
feature_cols = ['tx_amount', 'customer_avg_amount', 'amount_vs_avg_ratio', 'terminal_risk_count', 'tx_hour']

# Use the same test set (last 20%) as train_model.py, for a fair comparison
split_index = int(len(df) * 0.8)
X_test = df[feature_cols].iloc[split_index:]
y_test = df['tx_fraud'].iloc[split_index:]

# Get fraud probabilities for every test transaction
probabilities = model.predict_proba(X_test)[:, 1]

# Try a range of thresholds and record precision/recall at each
results = []
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    predictions = (probabilities >= threshold).astype(int)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    flagged = predictions.sum()
    results.append({
        'threshold': threshold,
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'transactions_flagged': int(flagged)
    })

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv('threshold_analysis.csv', index=False)
print("\nSaved threshold_analysis.csv")