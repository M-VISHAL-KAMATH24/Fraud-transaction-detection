from confluent_kafka import Producer
import json
import time
import random

# Kafka config
producer = Producer({'bootstrap.servers': 'localhost:9092'})

# Counter for evolving patterns
send_count = 0

# Generate mock transaction with evolving fraud patterns
def generate_mock_transaction():
    global send_count
    send_count += 1

    # Phase 1: Mostly normal (first 50 sends)
    if send_count < 50:
        return {
            'type': random.choice(['PAYMENT', 'CASH_IN', 'DEBIT']),
            'amount': random.uniform(10, 10000),  # Low amounts
            'oldbalanceOrg': random.uniform(1000, 100000),
            'newbalanceOrig': random.uniform(1000, 100000),
            'oldbalanceDest': random.uniform(0, 100000),
            'newbalanceDest': random.uniform(0, 100000),
            'step': random.randint(1, 743),
            'isFlaggedFraud': 0
        }
    # Phase 2: Evolving to more fraud-like (higher amounts, TRANSFERs)
    else:
        return {
            'type': 'TRANSFER' if random.random() < 0.7 else 'CASH_OUT',  # More suspicious types
            'amount': random.uniform(50000, 1000000),  # High amounts
            'oldbalanceOrg': 0 if random.random() < 0.6 else random.uniform(0, 1000),  # Often zero for fraud
            'newbalanceOrig': 0,
            'oldbalanceDest': random.uniform(0, 100000),
            'newbalanceDest': random.uniform(0, 100000),
            'step': random.randint(1, 743),
            'isFlaggedFraud': 1 if random.random() < 0.5 else 0  # 50% flagged
        }

# Send loop
while True:
    transaction = generate_mock_transaction()
    producer.produce('fraud_transactions', value=json.dumps(transaction))
    producer.flush()
    print(f"Sent transaction #{send_count}: {transaction}")
    time.sleep(2)  # Simulate real-time
