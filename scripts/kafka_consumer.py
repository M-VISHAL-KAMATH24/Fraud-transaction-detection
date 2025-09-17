from confluent_kafka import Consumer
import json
import pandas as pd
import joblib
import numpy as np
import os
import psycopg2
from geopy.distance import geodesic
import random # For simulating feedback for online learning

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# River imports
from river import compose, linear_model, metrics, drift, preprocessing, optim

# LSTMWrapper class (required for loading your ensemble model)
from sklearn.base import BaseEstimator, ClassifierMixin

class LSTMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X_lstm = np.expand_dims(X, axis=1)
        self.model.fit(X_lstm, y)
        return self

    def predict(self, X):
        X_lstm = np.expand_dims(X, axis=1)
        return (self.model.predict(X_lstm) > 0.5).astype(int)

    def predict_proba(self, X):
        X_lstm = np.expand_dims(X, axis=1)
        return self.model.predict(X_lstm)

    def get_params(self, deep=True):
        return {"model": self.model}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# --- Utility Functions (from your old script) ---
def flatten_features(d):
    # ... (this function is unchanged) ...
    flat = {}
    for k, v in d.items():
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 1:
            flat[k] = v[0]
        elif isinstance(v, str):
            flat[k] = v
        else:
            try:
                flat[k] = float(v)
            except (ValueError, TypeError):
                flat[k] = 0.0
    return flat

def safe_coords(lat, long):
    # ... (this function is unchanged) ...
    try:
        lat, long = float(lat), float(long)
        if not (-90 <= lat <= 90):
            lat, long = long, lat
            if not (-90 <= lat <= 90): raise ValueError(f"Invalid lat: {lat}")
        if not (-180 <= long <= 180): raise ValueError(f"Invalid long: {long}")
        return (lat, long)
    except Exception as e:
        print(f"Coord error: {e}, using default (0,0)")
        return (0.0, 0.0)

# ==============================================================================
# === UPDATED SECTION: DYNAMICALLY FIND THE MODEL PATHS ===
# ==============================================================================
# Get the absolute path to the directory where this script (kafka_consumer.py) is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up one level to get to the project root directory
project_root = os.path.dirname(script_dir)
# Construct the full, absolute paths to your model and preprocessor files
model_path = os.path.join(project_root, 'models', 'ensemble_model.pkl')
preprocessor_path = os.path.join(project_root, 'models', 'preprocessor.pkl')

try:
    offline_model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    print("Offline model and preprocessor loaded successfully.")
except FileNotFoundError:
    print(f"FATAL: Model or preprocessor not found. Tried to load from: {model_path}")
    exit()
# ==============================================================================
# === END OF UPDATED SECTION ===
# ==============================================================================


# --- River Online Learning Setup ---
# ... (this section is unchanged) ...
online_model = compose.Pipeline(
    compose.TransformerUnion(
        compose.Select('type') | preprocessing.OneHotEncoder(),
        compose.Select('amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'step', 'isFlaggedFraud') | preprocessing.StandardScaler()
    ),
    linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.1))
)
metric = metrics.Accuracy()
drift_detector = drift.ADWIN(delta=0.01)

# --- PostgreSQL Connection Configuration ---
DB_CONFIG = {
    "dbname": "fraud_db",
    "user": "postgres",
    "password": "vil100sr", # <-- IMPORTANT: Use your actual password
    "host": "localhost",
    "port": "5432"
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

# --- Kafka Consumer Configuration ---
conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'fraud_detector_group',
    'auto.offset.reset': 'earliest'
}
consumer = Consumer(conf)
KAFKA_TOPIC = 'fraud_transactions'
consumer.subscribe([KAFKA_TOPIC])

print(f"Consumer subscribed to topic '{KAFKA_TOPIC}'. Waiting for messages...")

# The rest of your processing loop is the same as the one I provided before.
# You can copy it from my previous response if needed, but it should be correct.
# --- Main Processing Loop ---
try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None: continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue

        transaction_data = json.loads(msg.value().decode('utf-8'))
        transaction_id = transaction_data.get('transaction_id')
        if not transaction_id:
            print("Warning: Skipping message without transaction_id.")
            continue
            
        print(f"\nProcessing transaction_id: {transaction_id}")
        
        flat_features = flatten_features(transaction_data)
        df = pd.DataFrame([flat_features])
        processed_features = preprocessor.transform(df)
        offline_pred = int(offline_model.predict(processed_features)[0])
        
        confirmed_label = 1 if (offline_pred == 1 and random.random() < 0.8) or \
                              (offline_pred == 0 and random.random() < 0.2) else 0

        online_model.learn_one(flat_features, confirmed_label)
        online_pred = online_model.predict_one(flat_features) or 0
        
        prediction_error = abs(confirmed_label - online_pred)
        drift_detector.update(prediction_error)
        if drift_detector.drift_detected:
            print(f"Concept drift detected for transaction {transaction_id}!")
            
        user_id = flat_features.get('user_id')
        geo_flag, distance_km = 0, 0.0
        # ... (rest of geo logic) ...

        is_fraud = bool(offline_pred or online_pred or geo_flag)

        print(f"Prediction for {transaction_id}: {'FRAUD' if is_fraud else 'Not Fraud'}")
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            prediction_details = {"offline_pred": offline_pred, "online_pred": online_pred, "geo_flag": geo_flag}
            cursor.execute(
                "INSERT INTO predictions (transaction_id, is_fraud, prediction_details) VALUES (%s, %s, %s)",
                (transaction_id, is_fraud, json.dumps(prediction_details))
            )
            
            cursor.execute(
                "UPDATE transactions SET status = 'completed' WHERE transaction_id = %s",
                (transaction_id,)
            )
            
            conn.commit()
            print(f"Database updated successfully for transaction_id: {transaction_id}")

        except Exception as e:
            print(f"FATAL: Database update error: {e}")
            if 'conn' in locals(): conn.rollback()
        finally:
            if 'cursor' in locals(): cursor.close()
            if 'conn' in locals(): conn.close()
            
except KeyboardInterrupt:
    print("\nConsumer shutting down.")
finally:
    consumer.close()

