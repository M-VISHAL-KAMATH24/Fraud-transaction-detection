from confluent_kafka import Consumer
import json
import pandas as pd
import joblib
import numpy as np
import psycopg2
import os
import time
import math

# ANSI color codes for impressive terminal output
class colors:
    HEADER = '\033[95m'; OKBLUE = '\033[94m'; OKGREEN = '\033[92m'
    WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'

# LSTMWrapper class (unchanged)
from sklearn.base import BaseEstimator, ClassifierMixin
class LSTMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model): self.model = model
    def fit(self, X, y): return self
    def predict(self, X):
        X_lstm = np.expand_dims(X, axis=1); return (self.model.predict(X_lstm) > 0.5).astype(int)
    def predict_proba(self, X):
        X_lstm = np.expand_dims(X, axis=1); return self.model.predict(X_lstm)
    def get_params(self, deep=True): return {"model": self.model}
    def set_params(self, **params):
        for k, v in params.items(): setattr(self, k, v); return self

# Load models
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'ensemble_model.pkl')
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'preprocessor.pkl')
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# DB Config
DB_CONFIG = {"dbname": "fraud_db", "user": "postgres", "password": "vil100sr", "host": "localhost", "port": "5432"}
def get_db_connection(): return psycopg2.connect(**DB_CONFIG)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371; lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 2 * math.asin(math.sqrt(a)) * R

# Kafka config
consumer = Consumer({
    'bootstrap.servers': 'localhost:9092', 'group.id': 'fraud_detector_legacy_final',
    'auto.offset.reset': 'earliest', 'enable.auto.commit': False
})
consumer.subscribe(['fraud_transactions'])

print(f"{colors.HEADER}--- Legacy Consumer Ready (Final Presentation Mode) ---{colors.ENDC}")
geo_threshold_km, frequent_tx_threshold, frequent_tx_window_mins = 500, 5, 2

while True:
    msg = consumer.poll(1.0)
    if msg is None: continue
    if msg.error(): print(f"{colors.FAIL}Error: {msg.error()}{colors.ENDC}"); continue

    conn, cursor = None, None
    try:
        data = json.loads(msg.value().decode('utf-8'))
        tx_id, user_id = data.get('transaction_id'), data.get('user_id')
        amount, oldbalanceOrg = data.get('amount', 0.0), data.get('oldbalanceOrg', 0.0)
        
        print(f"\n{colors.HEADER}==================================================================={colors.ENDC}")
        print(f"{colors.BOLD}Received Legacy TX:{colors.ENDC} {tx_id} | {colors.BOLD}User:{colors.ENDC} {user_id}")
        
        conn = get_db_connection()
        cursor = conn.cursor()

        # --- Rule 1: Frequent Transaction ---
        cursor.execute("SELECT COUNT(*) FROM transactions_legacy_demo WHERE user_id = %s AND created_at >= (NOW() - INTERVAL '%s minutes')", (user_id, frequent_tx_window_mins))
        freq_count = cursor.fetchone()[0]
        freq_fraud = freq_count >= frequent_tx_threshold

        # --- Rule 2: Geo Distance ---
        distance = haversine(data.get('billing_lat',0), data.get('billing_long',0), data.get('tx_lat',0), data.get('tx_long',0))
        geo_fraud = distance > geo_threshold_km

        # --- Rule 3: Full Drain Attempt ---
        drain_fraud = oldbalanceOrg > 0 and abs(amount - oldbalanceOrg) < 1

        # --- Rule 4: ML Model ---
        df = pd.DataFrame([data]); processed = preprocessor.transform(df)
        ml_pred = model.predict(processed)[0]
        
        # Combine rules for final verdict
        is_fraud, reasons = False, []
        if freq_fraud: is_fraud = True; reasons.append(f"Frequent Transactions ({freq_count + 1} in {frequent_tx_window_mins}m)")
        if geo_fraud: is_fraud = True; reasons.append(f"Geo Anomaly ({distance:,.0f}km)")
        if drain_fraud: is_fraud = True; reasons.append("Full Account Drain")
        if ml_pred == 1 and not is_fraud: is_fraud = True; reasons.append("Suspicious Pattern (ML Model)")

        # Display verdict
        if is_fraud:
            print(f"{colors.FAIL}{colors.BOLD}>>> FRAUD DETECTED <<<{colors.ENDC}")
            print(f"  - {colors.WARNING}{colors.BOLD}Reason(s):{colors.ENDC} {', '.join(reasons)}")
        else:
            print(f"{colors.OKGREEN}{colors.BOLD}>>> TRANSACTION OK <<<{colors.ENDC}")

        # Display details
        print(f"  - {colors.BOLD}Amount:{colors.ENDC} ${amount:,.2f} | {colors.BOLD}Type:{colors.ENDC} {data.get('type')}")
        print(f"  - {colors.BOLD}Geo Distance:{colors.ENDC} {distance:,.0f} km | {colors.BOLD}Old Balance:{colors.ENDC} ${oldbalanceOrg:,.2f}")

        # Save to DB
        cursor.execute("INSERT INTO transactions_legacy_demo (user_id, transaction_data, status) VALUES (%s, %s, %s);", (user_id, json.dumps(data), 'completed'))
        cursor.execute("INSERT INTO predictions_legacy_demo (transaction_id, is_fraud, prediction_details) VALUES (%s, %s, %s);", (tx_id, is_fraud, json.dumps({'rule': ', '.join(reasons) if reasons else 'None'})))
        conn.commit()
        print(f"{colors.OKBLUE}  - Record saved successfully to 'predictions_legacy_demo'.{colors.ENDC}")

        consumer.commit(message=msg)

    except Exception as e:
        print(f"{colors.FAIL}--- ERROR processing message: {e} ---{colors.ENDC}")
        if conn: conn.rollback()
    finally:
        if cursor: cursor.close()
        if conn: conn.close()
