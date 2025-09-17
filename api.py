from flask import Flask, request, jsonify
from confluent_kafka import Producer
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import socket # For Kafka client.id

app = Flask(__name__)

# --- Kafka Producer Configuration ---
conf = {
    'bootstrap.servers': 'localhost:9092',
    'client.id': socket.gethostname()
}
producer = Producer(conf)
KAFKA_TOPIC = 'fraud_transactions' # Use the same topic as your consumer

# --- PostgreSQL Connection Configuration ---
DB_CONFIG = {
    "dbname": "fraud_db",
    "user": "postgres",
    "password": "vil100sr", # <-- IMPORTANT: Use your actual password
    "host": "localhost",
    "port": "5432"
}

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    return psycopg2.connect(**DB_CONFIG)

@app.route('/submit_transaction', methods=['POST'])
def submit_transaction():
    """
    Receives a transaction, stores it in Postgres, and produces it to Kafka.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    # --- Step 1: Insert the transaction into PostgreSQL ---
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # We store the full JSON payload for auditing and retraining
        cursor.execute(
            """
            INSERT INTO transactions (user_id, transaction_data, status)
            VALUES (%s, %s, %s) RETURNING transaction_id;
            """,
            (data.get('user_id'), json.dumps(data), 'pending')
        )
        transaction_id = cursor.fetchone()[0]
        conn.commit()
    except Exception as e:
        print(f"Database Error: {e}")
        return jsonify({"error": "Failed to store transaction"}), 500
    finally:
        cursor.close()
        conn.close()

    # --- Step 2: Add transaction_id to payload and send to Kafka ---
    data['transaction_id'] = transaction_id # Crucial for linking prediction back
    
    try:
        producer.produce(KAFKA_TOPIC, value=json.dumps(data))
        producer.flush() # Ensure message is sent
    except Exception as e:
        print(f"Kafka Error: {e}")
        return jsonify({"error": "Failed to send transaction to processing queue"}), 500

    # --- Step 3: Return the ID to the client ---
    return jsonify({
        "message": "Transaction submitted for processing.",
        "transaction_id": transaction_id
    }), 202 # 202 Accepted means the request is accepted but processing is not complete

# You will add the /get_prediction endpoint in the next task
# For now, this is enough to get started.
@app.route('/get_prediction/<int:transaction_id>', methods=['GET'])
def get_prediction(transaction_id):
    """
    Checks the database for the prediction result of a given transaction.
    """
    try:
        conn = get_db_connection()
        # Use RealDictCursor to get results as dictionaries
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Query the predictions table
        cursor.execute(
            "SELECT is_fraud, prediction_details FROM predictions WHERE transaction_id = %s", 
            (transaction_id,)
        )
        prediction = cursor.fetchone()
        
        if prediction:
            # Result is found
            return jsonify({
                "status": "completed",
                "is_fraud": prediction['is_fraud'],
                "details": prediction['prediction_details']
            })
        else:
            # Result not yet in the database
            return jsonify({"status": "pending"}), 202

    except Exception as e:
        print(f"Database error on get_prediction: {e}")
        return jsonify({"error": "Failed to fetch prediction"}), 500
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

