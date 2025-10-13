from confluent_kafka import Consumer
import json
import pandas as pd
import joblib
import numpy as np
import psycopg2 # NEW: Import database driver

# Define LSTMWrapper class (unchanged)
from sklearn.base import BaseEstimator, ClassifierMixin
class LSTMWrapper(BaseEstimator, ClassifierMixin):
    # ... (the entire LSTMWrapper class is unchanged)
    pass

# Load Week 1 models (unchanged)
model = joblib.load('models/ensemble_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# --- NEW: Database Configuration ---
DB_CONFIG = {"dbname": "fraud_db", "user": "postgres", "password": "vil100sr", "host": "localhost", "port": "5432"}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

# Kafka config (Manual commit enabled for reliability)
consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'fraud_detector_legacy', # Use a unique group.id
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': False
})
consumer.subscribe(['fraud_transactions'])

print("--- Legacy Consumer Ready ---")
# Process loop
while True:
    msg = consumer.poll(1.0)
    if msg is None: continue
    if msg.error():
        print(f"Error: {msg.error()}")
        continue

    conn, cursor = None, None
    try:
        data = json.loads(msg.value().decode('utf-8'))
        tx_id = data.get('transaction_id')
        user_id = data.get('user_id')
        
        # Preprocess and predict (unchanged)
        df = pd.DataFrame([data])
        processed = preprocessor.transform(df)
        prediction = model.predict(processed)[0]
        
        print(f"\nReceived Legacy TX: {tx_id} | User: {user_id}")
        print(f"--> Prediction: {'FRAUD' if prediction == 1 else 'Not Fraud'} (Model: LSTM_V1)")

        # --- NEW: Save results to DEMO tables ---
        conn = get_db_connection()
        cursor = conn.cursor()

        # Insert raw transaction data for records
        cursor.execute(
            "INSERT INTO transactions_legacy_demo (user_id, transaction_data, status) VALUES (%s, %s, %s);",
            (user_id, json.dumps(data), 'completed')
        )

        # Insert the prediction result
        cursor.execute(
            "INSERT INTO predictions_legacy_demo (transaction_id, is_fraud, prediction_details) VALUES (%s, %s, %s);",
            (tx_id, bool(prediction), json.dumps({"rule": "LSTM_MODEL_V1"}))
        )
        
        conn.commit()
        print(f"--> Results for {tx_id} saved to demo tables.")

        # Manually commit to Kafka AFTER DB write is successful
        consumer.commit(message=msg)

    except Exception as e:
        print(f"--- ERROR processing legacy message: {e} ---")
        if conn: conn.rollback()
    finally:
        if cursor: cursor.close()
        if conn: conn.close()
