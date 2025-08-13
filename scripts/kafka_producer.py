from confluent_kafka import Producer
import json
import time
import random

# Kafka config (adjust bootstrap.servers if not local)
producer = Producer({'bootstrap.servers': 'localhost:9092'})

# Generate mock transaction (based on PaySim features)
def generate_mock_transaction():
    return {
        'type': random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT']),
        'amount': random.uniform(10, 1000000),
        'oldbalanceOrg': random.uniform(0, 1000000),
        'newbalanceOrig': random.uniform(0, 1000000),
        'oldbalanceDest': random.uniform(0, 1000000),
        'newbalanceDest': random.uniform(0, 1000000),
        'step': random.randint(1, 743),  # PaySim range
        'isFlaggedFraud': random.choice([0, 1])
    }

# Send loop (runs forever; Ctrl+C to stop)
while True:
    transaction = generate_mock_transaction()
    producer.produce('fraud_transactions', value=json.dumps(transaction))
    producer.flush()
    print(f"Sent transaction: {transaction}")
    time.sleep(2)  # Adjust for faster/slower simulation
