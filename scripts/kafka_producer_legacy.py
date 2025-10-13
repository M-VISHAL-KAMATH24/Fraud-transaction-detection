from confluent_kafka import Producer
import json
import time
import random

# Kafka config
producer = Producer({'bootstrap.servers': 'localhost:9092'})

# Counter for evolving patterns and unique IDs
send_count = 0

# Hardcoded list of user_ids (unchanged)
db_user_ids = [
    'user_1', 'user_2', 'user_3', 'user_4', 'user_5',
    'user_6', 'user_7', 'user_8', 'user_9', 'user_10',
    'user_11', 'user_12', 'user_13', 'user_14', 'user_15'
]

# Corresponding geo data for users (unchanged)
user_geo = {
    'user_1': {'billing_country': 'IN', 'billing_lat': 28.6139, 'billing_long': 77.2090},
    'user_2': {'billing_country': 'US', 'billing_lat': 37.7749, 'billing_long': -122.4194},
    # ... (rest of the user_geo dictionary is unchanged)
    'user_15': {'billing_country': 'AU', 'billing_lat': -37.8136, 'billing_long': 144.9631},
}

def generate_mock_transaction():
    global send_count
    send_count += 1
    
    # --- All the logic to generate the transaction is unchanged ---
    user_id = random.choice(db_user_ids)
    billing_info = user_geo.get(user_id, {})
    billing_country = billing_info.get('billing_country', 'N/A')
    billing_lat = billing_info.get('billing_lat', 0.0)
    billing_long = billing_info.get('billing_long', 0.0)

    tx_country, tx_lat, tx_long = billing_country, billing_lat, billing_long
    if random.random() < 0.4:
        tx_country = random.choice(['RU', 'NG', 'CN'])
        tx_lat += random.uniform(10, 20)
        tx_long += random.uniform(10, 20)

    if send_count < 50:
        transaction_details = { 'type': random.choice(['PAYMENT', 'CASH_IN']), 'amount': random.uniform(10, 1000) }
    else:
        transaction_details = { 'type': 'TRANSFER', 'amount': random.uniform(50000, 1000000) }

    # --- NEW: Add a unique transaction_id ---
    # This is crucial for tracking in the database
    transaction_id = f"legacy-{int(time.time() * 1000)}-{send_count}"

    transaction = {
        'transaction_id': transaction_id, # Added this line
        'user_id': user_id,
        'billing_country': billing_country,
        'billing_lat': billing_lat,
        'billing_long': billing_long,
        'tx_country': tx_country,
        'tx_lat': tx_lat,
        'tx_long': tx_long,
        **transaction_details
    }
    return transaction

# Sending loop
while True:
    transaction = generate_mock_transaction()
    producer.produce('fraud_transactions', value=json.dumps(transaction))
    producer.flush()
    print(f"Sent LEGACY transaction: {transaction}")
    time.sleep(2)
