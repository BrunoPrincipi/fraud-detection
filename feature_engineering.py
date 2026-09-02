import pandas as pd
from sqlalchemy import create_engine

# Connect to your database
engine = create_engine('postgresql://brunoprincipi@localhost/fraud_detection')
df = pd.read_sql('SELECT * FROM transactions ORDER BY tx_datetime', engine)

# Feature 1: customer's average spend so far (rolling, using only past transactions)
df['customer_avg_amount'] = df.groupby('customer_id')['tx_amount'].transform(
    lambda x: x.expanding().mean().shift(1)
)

# Feature 2: how far this transaction's amount is from the customer's average (the "spike" signal)
df['amount_vs_avg_ratio'] = df['tx_amount'] / df['customer_avg_amount']

# Feature 3: transaction count per terminal in the last 28 days (mirrors scenario 2's window)
df['terminal_risk_count'] = df.groupby('terminal_id')['tx_fraud'].transform(
    lambda x: x.shift(1).rolling(window=100, min_periods=1).sum()
)

# Feature 4: hour of day (fraud may cluster at unusual hours)
df['tx_hour'] = pd.to_datetime(df['tx_datetime']).dt.hour

df.to_csv('features.csv', index=False)
print(df.shape)