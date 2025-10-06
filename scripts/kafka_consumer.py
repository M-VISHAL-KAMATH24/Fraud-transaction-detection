import json
import os
import joblib
import numpy as np
import pandas as pd
import psycopg2
from confluent_kafka import Consumer
from sklearn.base import BaseEstimator, ClassifierMixin

# --- Suppress TensorFlow/oneDNN warnings ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- LSTMWrapper for loading the model (unchanged) ---
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

# --- Paths and Model Loading ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
model_path = os.path.join(project_root, 'models', 'ensemble_model.pkl')
preprocessor_path = os.path.join(project_root, 'models', 'preprocessor.pkl')

try:
    offline_model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    print("Offline model and preprocessor loaded successfully.")
except FileNotFoundError:
    print(f"FATAL: Model or preprocessor not found at {model_path}")
    exit()

# --- DB and Kafka Configuration ---
DB_CONFIG = {"dbname": "fraud_db", "user": "postgres", "password": "vil100sr", "host": "localhost", "port": "5432"}
KAFKA_TOPIC = 'fraud_transactions'
conf = {'bootstrap.servers': 'localhost:9092', 'group.id': 'fraud_detector_group', 'auto.offset.reset': 'earliest'}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

consumer = Consumer(conf)
consumer.subscribe([KAFKA_TOPIC])
print(f"Consumer subscribed to topic '{KAFKA_TOPIC}'. Waiting for messages...")

# --- Main Processing Loop ---
try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None: continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue

        try:
            transaction_data = json.loads(msg.value().decode('utf-8'))
            transaction_id = transaction_data.get('transaction_id')
            if not transaction_id:
                continue

            print(f"\n--- Processing transaction_id: {transaction_id} ---")
            
            is_fraud = False
            rule_triggered = "None"
            
            # === THE DEFINITIVE FIX: DIRECT VALUE CHECKING ===
            # We directly get the values we need and convert them safely.
            amount_val = float(transaction_data.get('amount', 0))
            balance_val = float(transaction_data.get('oldbalanceOrg', -1))
            tx_type = transaction_data.get('type')

            # Rule 1: Check for complete account drain
            if tx_type == 'TRANSFER' and abs(amount_val - balance_val) < 0.01:
                is_fraud = True
                rule_triggered = "Complete Account Drain Rule"
            
            # If no hard rule was triggered, proceed to the AI model
            if not is_fraud:
                # The model needs a DataFrame, so we create it on the fly
                df_for_model = pd.DataFrame([transaction_data])
                processed_features = preprocessor.transform(df_for_model)
                model_prediction = int(offline_model.predict(processed_features)[0])
                if model_prediction == 1:
                    is_fraud = True
                    rule_triggered = "AI Model Flag"

            print(f"Result: {'FRAUD' if is_fraud else 'Not Fraud'} (Reason: {rule_triggered})")

            # --- Database Finalization Logic ---
            conn = get_db_connection()
            cursor = conn.cursor()
            
            if not is_fraud:
                # Update balances ONLY if the transaction is approved
                sender_id = transaction_data.get('user_id')
                receiver_id = transaction_data.get('receiver_id')
                if sender_id and receiver_id and amount_val > 0:
                    cursor.execute("UPDATE users SET current_balance = current_balance - %s WHERE user_id = %s;", (amount_val, sender_id))
                    cursor.execute("UPDATE users SET current_balance = current_balance + %s WHERE user_id = %s;", (amount_val, receiver_id))

            # Record the final decision in the database
            prediction_details = {"rule_triggered": rule_triggered}
            cursor.execute("INSERT INTO predictions (transaction_id, is_fraud, prediction_details) VALUES (%s, %s, %s);", (transaction_id, is_fraud, json.dumps(prediction_details)))
            cursor.execute("UPDATE transactions SET status = 'completed' WHERE transaction_id = %s;", (transaction_id,))
            
            conn.commit()
            cursor.close()
            conn.close()
            print("Database successfully updated.")

        except Exception as e:
            print(f"--- ERROR processing message: {e} ---")
            # If something goes wrong, we try to roll back any partial DB changes
            if 'conn' in locals() and conn:
                conn.rollback()

except KeyboardInterrupt:
    print("\nConsumer shutting down.")
finally:
    consumer.close()
