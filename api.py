from flask import Flask, request, jsonify
from confluent_kafka import Producer
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import socket

app = Flask(__name__)

# --- Kafka Producer Configuration ---
conf = {'bootstrap.servers': 'localhost:9092', 'client.id': socket.gethostname()}
producer = Producer(conf)
KAFKA_TOPIC = 'fraud_transactions'

# --- PostgreSQL Connection Configuration ---
DB_CONFIG = {
    "dbname": "fraud_db",
    "user": "postgres",
    "password": "vil100sr", # <-- IMPORTANT: Use your password
    "host": "localhost",
    "port": "5432"
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

# --- API Endpoint to fetch users for the UI ---
@app.route('/get_users', methods=['GET'])
def get_users():
    """Fetches all users from the database."""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # === THE FIX IS ON THIS LINE ===
    # We now also select the 'email' column
    cursor.execute("SELECT user_id, user_name, current_balance, email FROM users ORDER BY user_id;")
    
    users = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(users)


# --- The rest of your api.py file remains unchanged ---
@app.route('/submit_transaction', methods=['POST'])
def submit_transaction():
    data = request.get_json()
    sender_id = data.get('sender_id')
    receiver_id = data.get('receiver_id')
    amount = float(data.get('amount'))

    if not all([sender_id, receiver_id, amount]):
        return jsonify({"error": "Missing sender_id, receiver_id, or amount"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM users WHERE user_id IN (%s, %s);", (sender_id, receiver_id))
        users = {u['user_id']: u for u in cursor.fetchall()}

        sender = users.get(sender_id)
        receiver = users.get(receiver_id)

        if not sender or not receiver:
            return jsonify({"error": "Invalid sender or receiver ID"}), 404
        
        if float(sender['current_balance']) < amount:
            return jsonify({"error": "Insufficient balance"}), 400

        transaction_payload = {
            "type": "TRANSFER", "user_id": sender_id, "receiver_id": receiver_id,
            "amount": amount,
            "oldbalanceOrg": float(sender['current_balance']),
            "newbalanceOrig": float(sender['current_balance']) - amount,
            "oldbalanceDest": float(receiver['current_balance']),
            "newbalanceDest": float(receiver['current_balance']) + amount,
            "step": 1, "isFlaggedFraud": 0,
        }

        cursor.execute(
            "INSERT INTO transactions (user_id, transaction_data, status) VALUES (%s, %s, %s) RETURNING transaction_id;",
            (sender_id, json.dumps(transaction_payload), 'pending')
        )
        transaction_id = cursor.fetchone()['transaction_id']
        conn.commit()

        transaction_payload['transaction_id'] = transaction_id
        producer.produce(KAFKA_TOPIC, value=json.dumps(transaction_payload))
        producer.flush()

        return jsonify({"message": "Transaction submitted for fraud check.", "transaction_id": transaction_id}), 202

    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"error": "An internal error occurred."}), 500
    finally:
        if 'conn' in locals() and not conn.closed:
            cursor.close()
            conn.close()

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
