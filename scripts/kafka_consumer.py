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

# Load environment variables for email
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# --- Email and SMTP Configuration ---
SMTP_HOST = os.getenv('SMTP_HOST')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASS = os.getenv('SMTP_PASS')
SENDER_EMAIL = os.getenv('SENDER_EMAIL')

# --- Model Loading (unchanged) ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'final_fraud_model.pkl')
try:
    fraud_pipeline = joblib.load(MODEL_PATH)
    print("Final, unified fraud detection pipeline loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load the pipeline model. Details: {e}")
    exit()

# --- DB and Kafka Configuration (unchanged) ---
DB_CONFIG = {"dbname": "fraud_db", "user": "postgres", "password": "vil100sr", "host": "localhost", "port": "5432"}
KAFKA_TOPIC = 'fraud_transactions'
conf = {'bootstrap.servers': 'localhost:9092', 'group.id': 'fraud_detector_group', 'auto.offset.reset': 'earliest'}

def get_db_connection(): return psycopg2.connect(**DB_CONFIG)

# --- Email Sending Function (unchanged) ---
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

# --- Rule Configuration ---
FREQUENT_TX_COUNT = 5
FREQUENT_TX_WINDOW_MINUTES = 5

consumer = Consumer(conf)
consumer.subscribe([KAFKA_TOPIC])
print("Consumer ready to send email alerts.")

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
            
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT email FROM users WHERE user_id = %s", (tx_data['user_id'],))
            sender_email_row = cursor.fetchone()
            sender_email = sender_email_row[0] if sender_email_row else None

            # --- Rule 1: Frequent Transaction Check (NEW) ---
            cursor.execute(
                """
                SELECT COUNT(*) FROM transactions
                WHERE user_id = %s AND created_at >= (NOW() - INTERVAL %s)
                """,
                (tx_data['user_id'], f"{FREQUENT_TX_WINDOW_MINUTES} minutes")
            )
            freq_tx_count = cursor.fetchone()[0]

            if freq_tx_count >= FREQUENT_TX_COUNT:
                is_fraud = True
                rule_triggered = f"FREQUENT_TX_BLOCK: {freq_tx_count + 1} transactions in {FREQUENT_TX_WINDOW_MINUTES} mins"
                if sender_email:
                    subject = "Security Alert: Frequent Transactions Detected"
                    body = (
                        f"Dear User,\n\nWe have detected an unusually high number of transactions ({freq_tx_count + 1}) "
                        f"from your account in the last {FREQUENT_TX_WINDOW_MINUTES} minutes. For your security, "
                        "this transaction has been blocked. Please contact support if this was not you."
                    )
                    send_email_alert(sender_email, subject, body)

            cursor.close()
            conn.close()
            
            # --- Rule 2: Full Drain Check (Unchanged) ---
            if not is_fraud and tx_data.get('type') == 'TRANSFER' and abs(amount_val - balance_val) < 0.01:
                is_fraud = True
                rule_triggered = "BLOCK: Complete Account Drain"
                if sender_email:
                    subject = "Security Alert: Transaction Blocked"
                    body = (
                        f"Dear User,\n\nA transaction from your account was blocked because it attempted "
                        f"to transfer your entire balance of ${balance_val:,.2f}. Your account has been secured."
                    )
                    send_email_alert(sender_email, subject, body)
            
            # --- Rule 3: AI Model Check (Unchanged) ---
            if not is_fraud:
                features_for_prediction = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
                df = pd.DataFrame([tx_data])[features_for_prediction]
                prediction = fraud_pipeline.predict(df)
                if int(prediction[0]) == 1:
                    is_fraud = True
                    rule_triggered = "AI Model Flag"
                    if sender_email:
                        subject = "Security Alert: Suspicious Transaction Blocked"
                        body = (
                            f"Dear User,\n\nA transaction of ${amount_val:,.2f} from your account was blocked "
                            "due to suspicious activity detected by our system. No funds have been transferred."
                        )
                        send_email_alert(sender_email, subject, body)

            print(f"Result: {'FRAUD' if is_fraud else 'Not Fraud'} (Reason: {rule_triggered})")
            
            # --- DB Logic (Unchanged) ---
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
