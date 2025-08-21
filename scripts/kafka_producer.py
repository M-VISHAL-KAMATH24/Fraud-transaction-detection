from confluent_kafka import Producer
import json
import time
import random

# Kafka config
producer = Producer({'bootstrap.servers': 'localhost:9092'})

# Counter for evolving patterns
send_count = 0

# Hardcoded list of user_ids from your DB samples (to match consumer)
db_user_ids = [
    'user_1', 'user_2', 'user_3', 'user_4', 'user_5',
    'user_6', 'user_7', 'user_8', 'user_9', 'user_10',
    'user_11', 'user_12', 'user_13', 'user_14', 'user_15'
]

# Corresponding geo data for users in DB
user_geo = {
    'user_1': {'billing_country': 'IN', 'billing_lat': 28.6139, 'billing_long': 77.2090},
    'user_2': {'billing_country': 'US', 'billing_lat': 37.7749, 'billing_long': -122.4194},
    'user_3': {'billing_country': 'GB', 'billing_lat': 51.5074, 'billing_long': -0.1278},
    'user_4': {'billing_country': 'DE', 'billing_lat': 52.5200, 'billing_long': 13.4050},
    'user_5': {'billing_country': 'FR', 'billing_lat': 48.8566, 'billing_long': 2.3522},
    'user_6': {'billing_country': 'CA', 'billing_lat': 43.6510, 'billing_long': -79.3470},
    'user_7': {'billing_country': 'AU', 'billing_lat': -33.8688, 'billing_long': 151.2093},
    'user_8': {'billing_country': 'JP', 'billing_lat': 35.6895, 'billing_long': 139.6917},
    'user_9': {'billing_country': 'RU', 'billing_lat': 55.7558, 'billing_long': 37.6173},
    'user_10': {'billing_country': 'BR', 'billing_lat': -23.5505, 'billing_long': -46.6333},
    'user_11': {'billing_country': 'IN', 'billing_lat': 19.0760, 'billing_long': 72.8777},
    'user_12': {'billing_country': 'US', 'billing_lat': 40.7128, 'billing_long': -74.0060},
    'user_13': {'billing_country': 'GB', 'billing_lat': 55.9533, 'billing_long': -3.1883},
    'user_14': {'billing_country': 'IN', 'billing_lat': 13.0827, 'billing_long': 80.2707},
    'user_15': {'billing_country': 'AU', 'billing_lat': -37.8136, 'billing_long': 144.9631},
}

# Generate mock transaction with evolving fraud patterns + geo functionality
def generate_mock_transaction():
    global send_count
    send_count += 1

    # Select random user_id and geo
    user_id = random.choice(db_user_ids)
    billing_info = user_geo[user_id]
    billing_country = billing_info['billing_country']
    billing_lat = billing_info['billing_lat']
    billing_long = billing_info['billing_long']

    # Simulate transaction geo (close for normal, far for suspicious)
    tx_country = billing_country
    tx_lat = billing_lat + random.uniform(-0.1, 0.1)  # Nearby location for normal
    tx_long = billing_long + random.uniform(-0.1, 0.1)
    if random.random() < 0.4:  # 40% chance of geo mismatch, simulating fraud
        tx_country = random.choice(['RU', 'NG', 'CN'])
        tx_lat += random.uniform(10, 20)
        tx_long += random.uniform(10, 20)

    # Transaction details evolving complexity
    if send_count < 50:
        transaction_details = {
            'type': random.choice(['PAYMENT', 'CASH_IN', 'DEBIT']),
            'amount': random.uniform(10, 10000),
            'oldbalanceOrg': random.uniform(1000, 100000),
            'newbalanceOrig': random.uniform(1000, 100000),
            'oldbalanceDest': random.uniform(0, 100000),
            'newbalanceDest': random.uniform(0, 100000),
            'step': random.randint(1, 743),
            'isFlaggedFraud': 0
        }
    else:
        transaction_details = {
            'type': 'TRANSFER' if random.random() < 0.7 else 'CASH_OUT',
            'amount': random.uniform(50000, 1000000),
            'oldbalanceOrg': 0 if random.random() < 0.6 else random.uniform(0, 1000),
            'newbalanceOrig': 0,
            'oldbalanceDest': random.uniform(0, 100000),
            'newbalanceDest': random.uniform(0, 100000),
            'step': random.randint(1, 743),
            'isFlaggedFraud': 1 if random.random() < 0.9 else 0
        }
    
    # Compose full transaction dictionary
    transaction = {
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
    print(f"Sent transaction #{send_count}: {transaction}")
    time.sleep(2)  # simulate real time
