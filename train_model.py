import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import lightgbm as lgb
import joblib

# Step 1: Load data
df = pd.read_csv('features.csv')
print("Initial shape:", df.shape)

# Step 2: Drop rows with missing features (each customer's first transaction)
df = df.dropna(subset=['customer_avg_amount', 'amount_vs_avg_ratio', 'terminal_risk_count'])
print("After dropping NAs:", df.shape)

# Step 2b: Clean extreme ratio values (cap at 20x average, since scenario 3 fraud only multiplies by 5x)
df['amount_vs_avg_ratio'] = df['amount_vs_avg_ratio'].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['amount_vs_avg_ratio'])
df['amount_vs_avg_ratio'] = df['amount_vs_avg_ratio'].clip(upper=20)
print("After capping ratio, max value:", df['amount_vs_avg_ratio'].max())

# Step 3: Sort by time and select features
df = df.sort_values('tx_datetime')
feature_cols = ['tx_amount', 'customer_avg_amount', 'amount_vs_avg_ratio', 'terminal_risk_count', 'tx_hour']
X = df[feature_cols]
y = df['tx_fraud']

# Step 4: Time-aware train/test split (train on past, test on future)
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
print("Train size:", len(X_train), "Test size:", len(X_test))

# Step 5: Train LightGBM (no scaling needed for tree models)
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
lgb_model.fit(X_train, y_train)

# Step 6: Scale features, then train Logistic Regression baseline
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Note: sklearn emits RuntimeWarnings (overflow/divide-by-zero) here due to quasi-complete
# separation — tx_amount deterministically predicts fraud for scenario 1 (>220 = fraud),
# pushing logistic regression's coefficient for that feature very large. Results remain
# stable and correct across multiple solvers/regularization strengths tested; this is a
# known limitation of linear models on deterministic-threshold features, and is precisely
# the kind of pattern LightGBM handles natively via tree splits.

log_model = LogisticRegression(class_weight='balanced', max_iter=1000, C=0.01, solver='liblinear')
log_model.fit(X_train_scaled, y_train)

# Step 7: Evaluate both models
print(f"\n--- LightGBM ---")
y_pred = lgb_model.predict(X_test)
y_proba = lgb_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred, digits=3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 3))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

print(f"\n--- Logistic Regression ---")
y_pred = log_model.predict(X_test_scaled)
y_proba = log_model.predict_proba(X_test_scaled)[:, 1]
print(classification_report(y_test, y_pred, digits=3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 3))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Save the LightGBM model for later use in FastAPI
joblib.dump(lgb_model, 'fraud_model.pkl')
print("\nModel saved as fraud_model.pkl")