import json
import os
import psycopg2
from confluent_kafka import Consumer

# --- No AI model imports needed for this version ---

# --- Paths, Models, Config ---
DB_CONFIG = {"dbname": "fraud_db", "user": "postgres", "password": "vil100sr", "host": "localhost", "port": "5432"}
KAFKA_TOPIC = 'fraud_transactions'
conf = {'bootstrap.servers': 'localhost:9092', 'group.id': 'fraud_detector_group', 'auto.offset.reset': 'earliest'}

def get_db_connection(): return psycopg2.connect(**DB_CONFIG)

consumer = Consumer(conf)
consumer.subscribe([KAFKA_TOPIC])
print("Consumer ready in EMERGENCY MODE (AI Bypassed). Only hard-coded rules are active.")

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None or msg.error(): continue

        try:
            tx_data = json.loads(msg.value().decode('utf-8'))
            tx_id = tx_data.get('transaction_id')
            if not tx_id: continue
            
            print(f"\n--- Processing tx_id: {tx_id} (EMERGENCY MODE) ---")
            
            is_fraud = False
            rule_triggered = "None"
            
            amount_val = float(tx_data.get('amount', 0))
            balance_val = float(tx_data.get('oldbalanceOrg', 0))
            
            # --- ONLY THE HARD-CODED RULE IS CHECKED ---
            if tx_data.get('type') == 'TRANSFER' and abs(amount_val - balance_val) < 0.01:
                is_fraud = True
                rule_triggered = "BLOCK: Complete Account Drain"
            else:
                # All other transactions are approved
                rule_triggered = "AUTO-APPROVED (AI Bypassed)"

            print(f"Result: {'FRAUD' if is_fraud else 'Not Fraud'} (Reason: {rule_triggered})")

            # --- Database Finalization Logic ---
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
