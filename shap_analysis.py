import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Step 1: Load the trained model and the same features used for training
model = joblib.load('fraud_model.pkl')
df = pd.read_csv('features.csv')

# Step 2: Same cleaning as train_model.py, so the data matches what the model expects
df = df.dropna(subset=['customer_avg_amount', 'amount_vs_avg_ratio', 'terminal_risk_count'])
df['amount_vs_avg_ratio'] = df['amount_vs_avg_ratio'].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['amount_vs_avg_ratio'])
df['amount_vs_avg_ratio'] = df['amount_vs_avg_ratio'].clip(upper=20)

feature_cols = ['tx_amount', 'customer_avg_amount', 'amount_vs_avg_ratio', 'terminal_risk_count', 'tx_hour']

# Step 3: Take a smaller sample — SHAP is slow on millions of rows, 5000 is plenty to see patterns
sample = df.sample(n=5000, random_state=42)
X_sample = sample[feature_cols]

# Step 4: Create the SHAP explainer for tree models (fast, exact method for LightGBM)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Handle  SHAP output formats
if isinstance(shap_values, list):
    shap_values_fraud = shap_values[1]
else:
    shap_values_fraud = shap_values

# Step 5: Global summary — which features matter most, overall
shap.summary_plot(shap_values_fraud, X_sample, show=False)
plt.savefig('shap_summary.png', bbox_inches='tight')
plt.close()
print("Saved shap_summary.png")

# Step 6: Explain one single flagged transaction in detail
fraud_example = sample[sample['tx_fraud'] == 1].iloc[0]
idx = sample.index.get_loc(fraud_example.name)

print("\nExample fraudulent transaction:")
print(fraud_example[feature_cols])
print("\nSHAP values for this transaction (how much each feature pushed toward fraud):")
for feat, val in zip(feature_cols, shap_values[idx]):
    print(f"  {feat}: {val:.4f}")