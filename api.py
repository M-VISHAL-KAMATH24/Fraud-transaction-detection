from flask import Flask, request, jsonify
from confluent_kafka import Producer
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import socket
import os
import time
import secrets
import smtplib
import ssl
from email.mime.text import MIMEText
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# --- Kafka Producer Configuration (unchanged) ---
conf = {'bootstrap.servers': 'localhost:9092', 'client.id': socket.gethostname()}
producer = Producer(conf)
KAFKA_TOPIC = 'fraud_transactions'

# --- PostgreSQL Connection Configuration ---
DB_CONFIG = {"dbname": "fraud_db", "user": "postgres", "password": "vil100sr", "host": "localhost", "port": "5432"}

# --- NEW: SMTP and OTP Configuration ---
SMTP_HOST = os.getenv('SMTP_HOST')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASS = os.getenv('SMTP_PASS')
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
OTP_TTL_SECONDS = int(os.getenv('OTP_TTL_SECONDS', 300))

# In-memory store for OTPs for simplicity. For production, use a database or Redis.
# Format: { "user_id": {"otp": "123456", "expires_at": 1678886400} }
otp_store = {}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

# --- NEW: Email Sending Function ---
def send_email_alert(to_email: str, subject: str, body: str):
    """Send an email via configured SMTP if credentials exist."""
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

# --- API Endpoints ---

@app.route('/get_users', methods=['GET'])
def get_users():
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT user_id, user_name, current_balance, email FROM users ORDER BY user_id;")
    users = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(users)

# --- OTP Endpoints (unchanged) ---
@app.route('/request_otp', methods=['POST'])
def request_otp():
    data = request.get_json()
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    # Get the sender's email from the database
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT email FROM users WHERE user_id = %s", (user_id,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if not user or not user.get('email'):
        return jsonify({"error": "Email for user not found"}), 404

    # Generate a secure 6-digit OTP
    otp = f"{secrets.randbelow(1_000_000):06d}"
    expires_at = int(time.time()) + OTP_TTL_SECONDS
    otp_store[user_id] = {"otp": otp, "expires_at": expires_at}

    # Send the email
    email_body = f"Your one-time password (OTP) is: {otp}\nIt will expire in {OTP_TTL_SECONDS // 60} minutes."
    send_email_alert(to_email=user['email'], subject="Your Fraud Alert Verification Code", body=email_body)

    return jsonify({"message": f"OTP sent to {user['email']}"}), 200

@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    data = request.get_json()
    user_id = data.get('user_id')
    otp = data.get('otp')

    if not user_id or not otp:
        return jsonify({"error": "user_id and otp are required"}), 400

    stored_otp_data = otp_store.get(user_id)

    if not stored_otp_data:
        return jsonify({"error": "No OTP was requested for this user. Please request one first."}), 400

    if time.time() > stored_otp_data["expires_at"]:
        otp_store.pop(user_id, None) # Clean up expired OTP
        return jsonify({"error": "OTP has expired. Please request a new one."}), 400

    if stored_otp_data["otp"] != otp:
        return jsonify({"error": "Invalid OTP provided."}), 400

    # If OTP is correct, remove it so it can't be used again
    otp_store.pop(user_id, None)
    return jsonify({"message": "OTP verified successfully."}), 200

# --- Core Transaction Endpoint (unchanged) ---
@app.route('/submit_transaction', methods=['POST'])
def submit_transaction():
    # This endpoint's logic remains the same as before
    data = request.get_json()
    try:
        # ... (rest of your existing submit_transaction logic) ...
        # Create transaction payload
        transaction_payload = {
            "type": "TRANSFER",
            "user_id": data['sender_id'],
            "receiver_id": data['receiver_id'],
            "amount": float(data['amount']),
            "oldbalanceOrg": 1000,
            "newbalanceOrig": 900,
            "oldbalanceDest": 500,
            "newbalanceDest": 600,
            "step": 1,
            "isFlaggedFraud": 0
        }
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("INSERT INTO transactions (user_id, transaction_data, status) VALUES (%s, %s, %s) RETURNING transaction_id;", (data['sender_id'], json.dumps(transaction_payload), 'pending'))
        transaction_id = cursor.fetchone()['transaction_id']
        conn.commit()
        transaction_payload['transaction_id'] = transaction_id
        producer.produce(KAFKA_TOPIC, value=json.dumps(transaction_payload))
        producer.flush()
        return jsonify({"message": "Transaction submitted for fraud check.", "transaction_id": transaction_id}), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_prediction/<int:transaction_id>', methods=['GET'])
def get_prediction(transaction_id):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT is_fraud, prediction_details FROM predictions WHERE transaction_id = %s", (transaction_id,))
    prediction = cursor.fetchone()
    cursor.close()
    conn.close()
    if prediction:
        return jsonify({"status": "completed", "is_fraud": prediction['is_fraud'], "details": prediction['prediction_details']})
    else:
        return jsonify({"status": "pending"}), 202

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
