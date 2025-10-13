from confluent_kafka import Producer
import json
import time
import random

# --- Configurations ---
producer = Producer({'bootstrap.servers': 'localhost:9092'})
send_count = 0
db_user_ids = ['user_1', 'user_2', 'user_3', 'user_4', 'user_5', 'user_6', 'user_7', 'user_8', 'user_9', 'user_10', 'user_11', 'user_12', 'user_13', 'user_14', 'user_15']
user_geo = {
    'user_1': {'billing_lat': 28.61, 'billing_long': 77.20}, 'user_2': {'billing_lat': 37.77, 'billing_long': -122.41},
    'user_3': {'billing_lat': 51.50, 'billing_long': -0.12}, 'user_4': {'billing_lat': 52.52, 'billing_long': 13.40},
    'user_5': {'billing_lat': 48.85, 'billing_long': 2.35}, 'user_6': {'billing_lat': 43.65, 'billing_long': -79.34},
    'user_7': {'billing_lat': -33.86, 'billing_long': 151.20}, 'user_8': {'billing_lat': 35.68, 'billing_long': 139.69},
    'user_9': {'billing_lat': 55.75, 'billing_long': 37.61}, 'user_10': {'billing_lat': -23.55, 'billing_long': -46.63},
    'user_11': {'billing_lat': 19.07, 'billing_long': 72.87}, 'user_12': {'billing_lat': 40.71, 'billing_long': -74.00},
    'user_13': {'billing_lat': 55.95, 'billing_long': -3.18}, 'user_14': {'billing_lat': 13.08, 'billing_long': 80.27},
    'user_15': {'billing_lat': -37.81, 'billing_long': 144.96},
}

def generate_single_transaction(user_id, mode='normal'):
    global send_count
    send_count += 1
    
    billing_info = user_geo.get(user_id, {})
    billing_lat, billing_long = billing_info.get('billing_lat', 0.0), billing_info.get('billing_long', 0.0)

    # Default to a normal transaction
    tx_type, amount, old_balance = 'PAYMENT', random.uniform(10, 5000), random.uniform(10000, 50000)
    tx_lat, tx_long = billing_lat + random.uniform(-0.1, 0.1), billing_long + random.uniform(-0.1, 0.1)

    if mode == 'geo':
        tx_type = 'TRANSFER'; tx_lat, tx_long = billing_lat + random.uniform(20, 40), billing_long + random.uniform(20, 40)
    elif mode == 'drain':
        tx_type = 'TRANSFER'; amount = old_balance * 0.999
    
    transaction_id = f"legacy-{int(time.time() * 1000)}-{send_count}"

    return {
        'transaction_id': transaction_id, 'user_id': user_id, 'billing_lat': billing_lat,
        'billing_long': billing_long, 'tx_lat': tx_lat, 'tx_long': tx_long,
        'type': tx_type, 'amount': amount, 'oldbalanceOrg': old_balance, 'newbalanceOrig': old_balance - amount,
        'oldbalanceDest': 0, 'newbalanceDest': amount, 'step': 1, 'isFlaggedFraud': 1 if mode != 'normal' else 0
    }

# --- Main Automatic Loop ---
print("--- Starting Continuous & Randomized Legacy Producer ---")
while True:
    # Randomly choose which type of event to simulate
    simulation_type = random.choice(['normal', 'geo', 'drain', 'frequent'])

    if simulation_type == 'frequent':
        print("\n>>> SIMULATING: FREQUENT TRANSACTION FRAUD <<<")
        user_for_burst = random.choice(db_user_ids)
        for i in range(6):
            tx = generate_single_transaction(user_for_burst, 'normal')
            producer.produce('fraud_transactions', value=json.dumps(tx)); producer.flush()
            print(f"  -> Sent burst TX {i+1}/6 for {user_for_burst}: {tx['transaction_id']}")
            time.sleep(0.5)
    else:
        user_id = random.choice(db_user_ids)
        tx = generate_single_transaction(user_id, simulation_type)
        producer.produce('fraud_transactions', value=json.dumps(tx)); producer.flush()
        print(f"\nSent {simulation_type.upper()} TX for {user_id}: {tx['transaction_id']}")
    
    time.sleep(3) # Pause between simulations
