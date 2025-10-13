from confluent_kafka import Producer
import json
import time
import random

producer = Producer({'bootstrap.servers': 'localhost:9092'})
send_count = 0

# Your user and geo data (condensed for clarity)
db_user_ids = ['user_1', 'user_2', 'user_3', 'user_4', 'user_5', 'user_6', 'user_7', 'user_8', 'user_9', 'user_10', 'user_11', 'user_12', 'user_13', 'user_14', 'user_15']
user_geo = {
    'user_1': {'billing_lat': 28.6139, 'billing_long': 77.2090}, 'user_2': {'billing_lat': 37.7749, 'billing_long': -122.4194},
    'user_3': {'billing_lat': 51.5074, 'billing_long': -0.1278}, 'user_4': {'billing_lat': 52.5200, 'billing_long': 13.4050},
    'user_5': {'billing_lat': 48.8566, 'billing_long': 2.3522}, 'user_6': {'billing_lat': 43.6510, 'billing_long': -79.3470},
    'user_7': {'billing_lat': -33.8688, 'billing_long': 151.2093}, 'user_8': {'billing_lat': 35.6895, 'billing_long': 139.6917},
    'user_9': {'billing_lat': 55.7558, 'billing_long': 37.6173}, 'user_10': {'billing_lat': -23.5505, 'billing_long': -46.6333},
    'user_11': {'billing_lat': 19.0760, 'billing_long': 72.8777}, 'user_12': {'billing_lat': 40.7128, 'billing_long': -74.0060},
    'user_13': {'billing_lat': 55.9533, 'billing_long': -3.1883}, 'user_14': {'billing_lat': 13.0827, 'billing_long': 80.2707},
    'user_15': {'billing_lat': -37.8136, 'billing_long': 144.9631},
}

def generate_mock_transaction():
    global send_count
    send_count += 1
    
    user_id = random.choice(db_user_ids)
    billing_info = user_geo.get(user_id, {})
    billing_lat, billing_long = billing_info.get('billing_lat', 0.0), billing_info.get('billing_long', 0.0)

    # --- Enhanced Fraud Simulation ---
    is_fraud_attempt = random.random() < 0.5  # 50% chance of being a fraud attempt

    if not is_fraud_attempt:
        # LEGITIMATE TRANSACTION: Small amount, nearby location
        tx_type = random.choice(['PAYMENT', 'CASH_IN', 'DEBIT'])
        amount = random.uniform(10, 5000)
        old_balance = random.uniform(amount, 100000)
        new_balance = old_balance - amount
        tx_lat, tx_long = billing_lat + random.uniform(-0.1, 0.1), billing_long + random.uniform(-0.1, 0.1)
    else:
        # FRAUDULENT TRANSACTION: Large amount, drains account, distant location
        tx_type = random.choice(['TRANSFER', 'CASH_OUT'])
        old_balance = random.uniform(10000, 50000)
        amount = old_balance * random.uniform(0.95, 1.0)
        new_balance = old_balance - amount
        tx_lat, tx_long = billing_lat + random.uniform(20, 40), billing_long + random.uniform(20, 40)

    transaction_id = f"legacy-{int(time.time() * 1000)}-{send_count}"

    transaction = {
        'transaction_id': transaction_id, 'user_id': user_id, 'billing_lat': billing_lat,
        'billing_long': billing_long, 'tx_lat': tx_lat, 'tx_long': tx_long,
        'type': tx_type, 'amount': amount, 'oldbalanceOrg': old_balance,
        'newbalanceOrig': new_balance, 'oldbalanceDest': random.uniform(0, 10000),
        'newbalanceDest': random.uniform(0, 10000) + amount, 'step': random.randint(1, 100),
        'isFlaggedFraud': 1 if is_fraud_attempt else 0
    }
    return transaction

print("--- Starting Legacy Producer with Enhanced Fraud Simulation ---")
while True:
    transaction = generate_mock_transaction()
    producer.produce('fraud_transactions', value=json.dumps(transaction))
    producer.flush()
    
    fraud_status = "FRAUD" if transaction['isFlaggedFraud'] == 1 else "LEGIT"
    print(f"Sent TX: {transaction['transaction_id']} | User: {transaction['user_id']} | Amount: ${transaction['amount']:.2f} | Intent: {fraud_status}")
    time.sleep(2)

