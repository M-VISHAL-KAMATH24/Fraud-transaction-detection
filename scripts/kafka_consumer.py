import json
import os
import joblib
import pandas as pd
import psycopg2
from confluent_kafka import Consumer
import smtplib
import ssl
from email.mime.text import MIMEText
from dotenv import load_dotenv
import math

# Load environment variables for email
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# --- Configurations ---
SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SENDER_EMAIL = (
    os.getenv('SMTP_HOST'), int(os.getenv('SMTP_PORT', 587)), os.getenv('SMTP_USER'),
    os.getenv('SMTP_PASS'), os.getenv('SENDER_EMAIL')
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'final_fraud_model.pkl')
try:
    fraud_pipeline = joblib.load(MODEL_PATH)
    print("Final, unified fraud detection pipeline loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load the pipeline model. Details: {e}")
    exit()

DB_CONFIG = {"dbname": "fraud_db", "user": "postgres", "password": "vil100sr", "host": "localhost", "port": "5432"}
KAFKA_TOPIC = 'fraud_transactions'

# KAFKA CONFIGURATION WITH MANUAL COMMIT
conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'fraud_detector_group',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': False  # Disable automatic commits for reliability
}

# --- Rule Constants ---
FREQUENT_TX_COUNT = 5
FREQUENT_TX_WINDOW_MINUTES = 5
GEO_DISTANCE_THRESHOLD_KM = 500

# --- Helper Functions ---
def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def send_email_alert(to_email: str, subject: str, body: str):
    if not all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SENDER_EMAIL]):
        print("SMTP configuration is missing. Cannot send email.")
        return
    message = MIMEText(body, "plain")
    message["Subject"] = subject
    message["From"] = SENDER_EMAIL
    message["To"] = to_email
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SENDER_EMAIL, [to_email], message.as_string())
        print(f"Successfully sent alert email to {to_email}")
    except Exception as e:
        print(f"Failed to send email alert: {e}")

def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return c * 6371 # Radius of Earth in kilometers

# --- Main Consumer Logic ---
consumer = Consumer(conf)
consumer.subscribe([KAFKA_TOPIC])
print("Consumer ready. All rules active. Using MANUAL commits.")

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None or msg.error(): continue

        conn, cursor = None, None
        try:
            tx_data = json.loads(msg.value().decode('utf-8'))
            tx_id = tx_data.get('transaction_id')
            if not tx_id:
                consumer.commit(message=msg)
                continue

            print(f"\n--- Processing tx_id: {tx_id} ---")
            is_fraud, rule_triggered = False, "None"
            
            amount_val = float(tx_data.get('amount', 0))
            balance_val = float(tx_data.get('oldbalanceOrg', 0))
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT email FROM users WHERE user_id = %s", (tx_data['user_id'],))
            sender_email_row = cursor.fetchone()
            sender_email = sender_email_row[0] if sender_email_row else None

            # --- Rule 1: Frequent Transaction Check ---
            cursor.execute("SELECT COUNT(*) FROM transactions WHERE user_id = %s AND created_at >= (NOW() - INTERVAL '%s minutes')", (tx_data['user_id'], FREQUENT_TX_WINDOW_MINUTES))
            freq_tx_count = cursor.fetchone()[0]
            if freq_tx_count >= FREQUENT_TX_COUNT:
                is_fraud = True
                rule_triggered = f"FREQUENT_TX_BLOCK: {freq_tx_count + 1} txns in {FREQUENT_TX_WINDOW_MINUTES} mins"
                if sender_email:
                    send_email_alert(sender_email, "Security Alert: Frequent Transactions Detected", "...")

            # --- Rule 2: Geolocation Check ---
            if not is_fraud:
                billing_lat, billing_long = float(tx_data.get('billing_lat', 0.0)), float(tx_data.get('billing_long', 0.0))
                tx_lat, tx_long = float(tx_data.get('tx_lat', 0.0)), float(tx_data.get('tx_long', 0.0))
                if billing_lat != 0.0 and tx_lat != 0.0:
                    distance_km = haversine(billing_lat, billing_long, tx_lat, tx_long)
                    print(f"Geolocation distance: {distance_km:.2f} km")
                    if distance_km > GEO_DISTANCE_THRESHOLD_KM:
                        is_fraud = True
                        rule_triggered = f"GEO_DISTANCE_ANOMALY: {distance_km:.0f} km"
                        if sender_email:
                            send_email_alert(sender_email, "Security Alert: Suspicious Location Detected", "...")

            # --- Rule 3: Full Drain Check ---
            if not is_fraud and tx_data.get('type') == 'TRANSFER' and balance_val > 0 and abs(amount_val - balance_val) < 0.01:
                is_fraud = True
                rule_triggered = "BLOCK: Complete Account Drain"
                if sender_email:
                    send_email_alert(sender_email, "Security Alert: Transaction Blocked", "...")
            
            # --- Rule 4: AI Model Check ---
            if not is_fraud:
                features = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
                df = pd.DataFrame([tx_data])[features]
                prediction = fraud_pipeline.predict(df)
                if int(prediction[0]) == 1:
                    is_fraud = True
                    rule_triggered = "AI Model Flag"
                    if sender_email:
                        send_email_alert(sender_email, "Security Alert: Suspicious Transaction Blocked", "...")

            print(f"Result: {'FRAUD' if is_fraud else 'Not Fraud'} (Reason: {rule_triggered})")
            
            # --- FINAL DB LOGIC ---
            if not is_fraud:
                cursor.execute("UPDATE users SET current_balance = current_balance - %s WHERE user_id = %s;", (amount_val, tx_data['user_id']))
                cursor.execute("UPDATE users SET current_balance = current_balance + %s WHERE user_id = %s;", (amount_val, tx_data['receiver_id']))
            
            cursor.execute("INSERT INTO predictions (transaction_id, is_fraud, prediction_details) VALUES (%s, %s, %s);", (tx_id, is_fraud, json.dumps({"rule": rule_triggered})))
            cursor.execute("UPDATE transactions SET status = 'completed' WHERE transaction_id = %s;", (tx_id,))
            
            conn.commit()
            print(f"--- DB updated successfully for tx_id: {tx_id} ---")

            # Manually commit the Kafka offset AFTER the DB commit is successful
            consumer.commit(message=msg)
            print(f"--- Kafka offset committed for tx_id: {tx_id} ---")

        except Exception as e:
            print(f"--- ERROR processing message: {e} ---")
            if conn: conn.rollback()
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

except KeyboardInterrupt:
    print("\nConsumer shutting down.")
finally:
    consumer.close()
