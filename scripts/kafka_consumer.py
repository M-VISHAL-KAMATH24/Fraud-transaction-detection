import json
import os
import joblib
import pandas as pd
import psycopg2
from confluent_kafka import Consumer

# This is the path to the NEW, CORRECT model file you will create
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'final_fraud_model.pkl')

try:
    # We only load the ONE, UNIFIED pipeline file.
    fraud_pipeline = joblib.load(MODEL_PATH)
    print("Final, unified fraud detection pipeline loaded successfully.")
except Exception as e:
    print("--------------------------------------------------------------------")
    print(f"FATAL ERROR: Could not load the pipeline model at '{MODEL_PATH}'.")
    print("Please run the 'train_model.py' script first to generate it.")
    print(f"Details: {e}")
    print("--------------------------------------------------------------------")
    exit()

# --- DB and Kafka Configuration (unchanged) ---
DB_CONFIG = {"dbname": "fraud_db", "user": "postgres", "password": "vil100sr", "host": "localhost", "port": "5432"}
KAFKA_TOPIC = 'fraud_transactions'
conf = {'bootstrap.servers': 'localhost:9092', 'group.id': 'fraud_detector_group', 'auto.offset.reset': 'earliest'}

def get_db_connection(): return psycopg2.connect(**DB_CONFIG)

consumer = Consumer(conf)
consumer.subscribe([KAFKA_TOPIC])
print("Consumer ready with the new, correct pipeline.")

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None or msg.error(): continue

        try:
            tx_data = json.loads(msg.value().decode('utf-8'))
            tx_id = tx_data.get('transaction_id')
            if not tx_id: continue

            print(f"\n--- Processing tx_id: {tx_id} ---")
            is_fraud = False
            rule_triggered = "None"
            
            amount_val = float(tx_data.get('amount', 0))
            balance_val = float(tx_data.get('oldbalanceOrg', 0))

            # Rule 1: Hard-coded check for complete account drain
            if tx_data.get('type') == 'TRANSFER' and abs(amount_val - balance_val) < 0.01:
                is_fraud = True
                rule_triggered = "BLOCK: Complete Account Drain"
            
            # Rule 2: Use the unified AI pipeline for all other cases
            if not is_fraud:
                # The model was trained on specific columns. We must provide them.
                features_for_prediction = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
                df = pd.DataFrame([tx_data])[features_for_prediction]
                
                # Feed the RAW data directly to the pipeline. It handles everything.
                prediction = fraud_pipeline.predict(df)
                
                if int(prediction[0]) == 1:
                    is_fraud = True
                    rule_triggered = "AI Model Flag"

            print(f"Result: {'FRAUD' if is_fraud else 'Not Fraud'} (Reason: {rule_triggered})")
            
            # --- DB Logic (unchanged) ---
            conn = get_db_connection()
            cursor = conn.cursor()
            if not is_fraud:
                cursor.execute("UPDATE users SET current_balance = current_balance - %s WHERE user_id = %s;", (amount_val, tx_data['user_id']))
                cursor.execute("UPDATE users SET current_balance = current_balance + %s WHERE user_id = %s;", (amount_val, tx_data['receiver_id']))
            cursor.execute("INSERT INTO predictions (transaction_id, is_fraud, prediction_details) VALUES (%s, %s, %s);", (tx_id, is_fraud, json.dumps({"rule": rule_triggered})))
            cursor.execute("UPDATE transactions SET status = 'completed' WHERE transaction_id = %s;", (tx_id,))
            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            print(f"--- ERROR processing message: {e} ---")
except KeyboardInterrupt:
    print("\nConsumer shutting down.")
finally:
    consumer.close()
