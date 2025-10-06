from confluent_kafka import Consumer
import json
import pandas as pd
import joblib
import numpy as np
import os
import psycopg2
from geopy.distance import geodesic
import random

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from river import compose, linear_model, preprocessing, optim
from sklearn.base import BaseEstimator, ClassifierMixin

class LSTMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
        self.classes_ = None
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X_lstm = np.expand_dims(X, axis=1)
        self.model.fit(X_lstm, y)
        return self
    def predict(self, X):
        X_lstm = np.expand_dims(X, axis=1)
        return (self.model.predict(X_lstm) > 0.5).astype(int)
    def predict_proba(self, X):
        X_lstm = np.expand_dims(X, axis=1)
        return self.model.predict(X_lstm)
    def get_params(self, deep=True):
        return {"model": self.model}
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def flatten_features(d):
    flat = {}
    for k, v in d.items():
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 1:
            flat[k] = v[0]
        elif isinstance(v, str):
            flat[k] = v
        else:
            try:
                flat[k] = float(v)
            except (ValueError, TypeError):
                flat[k] = 0.0
    return flat

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
model_path = os.path.join(project_root, 'models', 'ensemble_model.pkl')
preprocessor_path = os.path.join(project_root, 'models', 'preprocessor.pkl')

try:
    offline_model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    print("Offline model and preprocessor loaded successfully.")
except FileNotFoundError:
    print(f"FATAL: Model or preprocessor not found. Tried to load from: {model_path}")
    exit()

DB_CONFIG = {
    "dbname": "fraud_db",
    "user": "postgres",
    "password": "vil100sr",
    "host": "localhost",
    "port": "5432"
}
KAFKA_TOPIC = 'fraud_transactions'
conf = {'bootstrap.servers': 'localhost:9092', 'group.id': 'fraud_detector_group', 'auto.offset.reset': 'earliest'}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

consumer = Consumer(conf)
consumer.subscribe([KAFKA_TOPIC])
print(f"Consumer subscribed to topic '{KAFKA_TOPIC}'. Waiting for messages...")

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None: continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue

        transaction_data = json.loads(msg.value().decode('utf-8'))
        transaction_id = transaction_data.get('transaction_id')
        if not transaction_id:
            continue

        print(f"\n[DEBUG] === Processing transaction_id: {transaction_id} ===")
        print(f"[DEBUG] Raw Kafka Message: {transaction_data}")
        
        flat_features = flatten_features(transaction_data)
        
        is_fraud = False
        rule_triggered = "None"
        
        try:
            amount_val = float(transaction_data.get('amount', 0))
            balance_val = float(transaction_data.get('oldbalanceOrg', -1))
            tx_type = transaction_data.get('type')

            # --- DIAGNOSTIC PRINT STATEMENTS ---
            print(f"[DEBUG] Rule Check Inputs: amount_val={amount_val}, balance_val={balance_val}, type={tx_type}")
            print(f"[DEBUG] Is Type 'TRANSFER'? {tx_type == 'TRANSFER'}")
            print(f"[DEBUG] Absolute difference: {abs(amount_val - balance_val)}")
            # --- END DIAGNOSTICS ---

            if (tx_type == 'TRANSFER' and abs(amount_val - balance_val) < 0.001):
                is_fraud = True
                rule_triggered = "Complete Account Drain"
        except (ValueError, TypeError) as e:
            print(f"[DEBUG] ERROR during rule check: {e}")
            pass
        
        if not is_fraud:
            print("[DEBUG] Hard rule not triggered. Proceeding to AI Model.")
            df = pd.DataFrame([flat_features])
            processed_features = preprocessor.transform(df)
            model_prediction = int(offline_model.predict(processed_features)[0])
            if model_prediction == 1:
                is_fraud = True
                rule_triggered = "AI Model Flag"

        print(f"[FINAL] Prediction for {transaction_id}: {'FRAUD' if is_fraud else 'Not Fraud'} (Reason: {rule_triggered})")

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            if not is_fraud:
                sender_id = transaction_data.get('user_id')
                receiver_id = transaction_data.get('receiver_id')
                amount = float(transaction_data.get('amount'))

                if sender_id and receiver_id and amount > 0:
                    cursor.execute("UPDATE users SET current_balance = current_balance - %s WHERE user_id = %s;", (amount, sender_id))
                    cursor.execute("UPDATE users SET current_balance = current_balance + %s WHERE user_id = %s;", (amount, receiver_id))
                    print(f"[DB] Balances updated for transaction {transaction_id}.")

            prediction_details = {"rule_triggered": rule_triggered}
            cursor.execute("INSERT INTO predictions (transaction_id, is_fraud, prediction_details) VALUES (%s, %s, %s);", (transaction_id, is_fraud, json.dumps(prediction_details)))
            cursor.execute("UPDATE transactions SET status = 'completed' WHERE transaction_id = %s;", (transaction_id,))
            conn.commit()
            print(f"[DB] Database finalized for transaction_id: {transaction_id}")

        except Exception as e:
            print(f"[DB] FATAL: Database update error: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()

except KeyboardInterrupt:
    print("\nConsumer shutting down.")
finally:
    consumer.close()
