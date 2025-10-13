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
import random

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# --- Configurations ---
conf = {'bootstrap.servers': 'localhost:9092', 'client.id': socket.gethostname()}
producer = Producer(conf)
KAFKA_TOPIC = 'fraud_transactions'
DB_CONFIG = {"dbname": "fraud_db", "user": "postgres", "password": "vil100sr", "host": "localhost", "port": "5432"}
SMTP_HOST = os.getenv('SMTP_HOST')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASS = os.getenv('SMTP_PASS')
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
OTP_TTL_SECONDS = int(os.getenv('OTP_TTL_SECONDS', 300))
otp_store = {}

# --- Helper Functions ---

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    return psycopg2.connect(**DB_CONFIG)

def send_email_smtp(to_email: str, subject: str, body: str):
    """Connects to the SMTP server and sends an email."""
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
        print(f"Successfully sent OTP email to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def simulate_transaction_location(billing_lat, billing_long, is_fraud=False):
    """Generates transaction coordinates dynamically."""
    if not is_fraud:
        lat_offset = random.uniform(-0.05, 0.05)
        long_offset = random.uniform(-0.05, 0.05)
        return billing_lat + lat_offset, billing_long + long_offset
    else:
        lat_offset = random.uniform(20, 50) * random.choice([-1, 1])
        long_offset = random.uniform(20, 50) * random.choice([-1, 1])
        return billing_lat + lat_offset, billing_long + long_offset

# --- API Endpoints ---

@app.route('/get_users', methods=['GET'])
def get_users():
    """Fetches all users along with their location data."""
    conn, cursor = None, None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT user_id, user_name, current_balance, email, billing_lat, billing_long FROM users ORDER BY user_id;")
        users = cursor.fetchall()
        return jsonify(users)
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@app.route('/request_otp', methods=['POST'])
def request_otp():
    """Generates and sends an OTP to a user's email."""
    data = request.get_json()
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT email FROM users WHERE user_id = %s", (user_id,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if not user or not user.get('email'):
        return jsonify({"error": "Email for user not found"}), 404
    
    otp = f"{secrets.randbelow(1_000_000):06d}"
    expires_at = int(time.time()) + OTP_TTL_SECONDS
    otp_store[user_id] = {"otp": otp, "expires_at": expires_at}
    
    email_body = f"Your one-time password (OTP) is: {otp}\nIt will expire in {OTP_TTL_SECONDS // 60} minutes."
    send_email_smtp(to_email=user['email'], subject="Your Fraud Alert Verification Code", body=email_body)
    
    return jsonify({"message": f"OTP sent to {user['email']}"}), 200

@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    """Verifies a user-provided OTP."""
    data = request.get_json()
    user_id = data.get('user_id')
    otp = data.get('otp')
    if not user_id or not otp:
        return jsonify({"error": "user_id and otp are required"}), 400
    
    stored_otp_data = otp_store.get(user_id)
    if not stored_otp_data:
        return jsonify({"error": "No OTP was requested for this user."}), 400
    
    if time.time() > stored_otp_data["expires_at"]:
        otp_store.pop(user_id, None)
        return jsonify({"error": "OTP has expired."}), 400
    
    if stored_otp_data["otp"] != otp:
        return jsonify({"error": "Invalid OTP provided."}), 400
    
    otp_store.pop(user_id, None)
    return jsonify({"message": "OTP verified successfully."}), 200

@app.route('/submit_transaction', methods=['POST'])
def submit_transaction():
    """Submits a new transaction and sends it to Kafka for processing."""
    data = request.get_json()
    conn, cursor = None, None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute("SELECT billing_lat, billing_long, current_balance FROM users WHERE user_id = %s", (data['sender_id'],))
        user_data = cursor.fetchone()
        
        billing_lat = float(user_data['billing_lat']) if user_data and user_data['billing_lat'] is not None else 0.0
        billing_long = float(user_data['billing_long']) if user_data and user_data['billing_long'] is not None else 0.0
        current_balance = float(user_data['current_balance']) if user_data and user_data['current_balance'] is not None else 0.0

        is_geo_fraud_simulation = random.random() < 0.3
        tx_lat, tx_long = simulate_transaction_location(billing_lat, billing_long, is_fraud=is_geo_fraud_simulation)
        
        if is_geo_fraud_simulation: print("--- SIMULATING GEO-FRAUD ---")
        else: print("--- Simulating valid location ---")

        transaction_payload = {
            "type": "TRANSFER", "user_id": data['sender_id'], "receiver_id": data['receiver_id'],
            "amount": float(data['amount']), "oldbalanceOrg": current_balance, "newbalanceOrig": current_balance - float(data['amount']),
            "oldbalanceDest": 0, "newbalanceDest": 0, "step": 1, "isFlaggedFraud": 0,
            "billing_lat": billing_lat, "billing_long": billing_long, "tx_lat": tx_lat, "tx_long": tx_long
        }

        cursor.execute(
            "INSERT INTO transactions (user_id, transaction_data, status) VALUES (%s, %s, %s) RETURNING transaction_id;",
            (data['sender_id'], json.dumps(transaction_payload), 'pending')
        )
        transaction_id = cursor.fetchone()['transaction_id']
        conn.commit()
        
        transaction_payload['transaction_id'] = transaction_id
        producer.produce(KAFKA_TOPIC, value=json.dumps(transaction_payload))
        producer.flush()

        return jsonify({"message": "Transaction submitted for fraud check.", "transaction_id": transaction_id}), 202

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@app.route('/get_prediction/<int:transaction_id>', methods=['GET'])
def get_prediction(transaction_id):
    """Fetches the final fraud prediction result for a transaction."""
    conn, cursor = None, None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT is_fraud, prediction_details FROM predictions WHERE transaction_id = %s", (transaction_id,))
        prediction = cursor.fetchone()
        
        if prediction:
            # If a prediction is found, return it as "completed"
            return jsonify({
                "status": "completed", 
                "is_fraud": prediction['is_fraud'], 
                "details": prediction['prediction_details']
            })
        else:
            # If no prediction is found yet, it's still pending
            return jsonify({"status": "pending"}), 202
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
