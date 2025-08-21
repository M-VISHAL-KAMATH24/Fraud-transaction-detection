from confluent_kafka import Producer
import json
import time
import random
import pandas as pd  # For loading mock users

# Load mock users from CSV (add this at the top)
mock_users = pd.read_csv('mock_users.csv')  # Assumes you ran generate_mock_users.py

# Kafka config
producer = Producer({'bootstrap.servers': 'localhost:9092'})

# Counter for evolving patterns
send_count = 0

# Generate mock transaction with evolving fraud patterns + geo functionality
def generate_mock_transaction():
    global send_count
    send_count += 1

    # Sample a random user from mock DB
    user = mock_users.sample(1).iloc[0]

    # Geo fields from user (billing) and simulated transaction
    billing_country = user['billing_country']
    billing_lat = user['billing_lat']
    billing_long = user['billing_long']

    # Simulate transaction geo (close for normal, far for fraud-like)
    tx_country = billing_country
    tx_lat = billing_lat + random.uniform(-0.1, 0.1)
    tx_long = billing_long + random.uniform(-0.1, 0.1)
    if random.random() < 0.4:  # 40% chance of geo mismatch (for more flags in testing)
        tx_country = random.choice(['RU', 'NG', 'CN'])
        tx_lat += random.uniform(10, 20)  # Larger offset for >100km
        tx_long += random.uniform(10, 20)

    # Original evolving logic (unchanged)
    if send_count < 50:
        # Phase 1: Mostly normal (first 50 sends)
        transaction_details = {
            'type': random.choice(['PAYMENT', 'CASH_IN', 'DEBIT']),
            'amount': random.uniform(10, 10000),  # Low amounts
            'oldbalanceOrg': random.uniform(1000, 100000),
            'newbalanceOrig': random.uniform(1000, 100000),
            'oldbalanceDest': random.uniform(0, 100000),
            'newbalanceDest': random.uniform(0, 100000),
            'step': random.randint(1, 743),
            'isFlaggedFraud': 0
        }
    else:
        # Phase 2: Evolving to more fraud-like (higher amounts, TRANSFERs)
        transaction_details = {
            'type': 'TRANSFER' if random.random() < 0.7 else 'CASH_OUT',  # More suspicious types
            'amount': random.uniform(50000, 1000000),  # High amounts
            'oldbalanceOrg': 0 if random.random() < 0.6 else random.uniform(0, 1000),  # Often zero for fraud
            'newbalanceOrig': 0,
            'oldbalanceDest': random.uniform(0, 100000),
            'newbalanceDest': random.uniform(0, 100000),
            'step': random.randint(1, 743),
            'isFlaggedFraud': 1 if random.random() < 0.5 else 0  # 50% flagged
        }

    # Combine original details with geo fields
    transaction = {
        'user_id': user['user_id'],  # Add user_id for consumer lookup
        'billing_country': billing_country,
        'billing_lat': billing_lat,
        'billing_long': billing_long,
        'tx_country': tx_country,
        'tx_lat': tx_lat,
        'tx_long': tx_long,
        **transaction_details  # Merge in the evolving transaction fields
    }

    return transaction

# Send loop (unchanged)
while True:
    transaction = generate_mock_transaction()
    producer.produce('fraud_transactions', value=json.dumps(transaction))
    producer.flush()
    print(f"Sent transaction #{send_count}: {transaction}")
    time.sleep(2)  # Simulate real-time
