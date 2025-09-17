from confluent_kafka import Consumer
import json
import pandas as pd
import joblib
import numpy as np
import random  # For simulating feedback
import os  # To suppress TensorFlow warnings
from geopy.distance import geodesic  # For geo distance calculation
import psycopg2  # New: For PostgreSQL connection

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# River imports for online learning, preprocessing, and drift detection
from river import compose  # For Pipeline and TransformerUnion
from river import linear_model  # LogisticRegression
from river import metrics  # Accuracy tracking
from river import drift  # ADWIN for concept drift
from river import preprocessing  # For OneHotEncoder and StandardScaler
from river import anomaly  # For HalfSpaceTrees if used
from river import optim  # Added: For custom optimizer with higher learning rate

# LSTMWrapper class (required for loading ensemble)
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

# Load offline ensemble model
offline_model = joblib.load('../models/ensemble_model.pkl')  # Adjust path if needed
preprocessor = joblib.load('../models/preprocessor.pkl')

# New: PostgreSQL connection (use your actual password; consider env vars for security)
conn = psycopg2.connect(dbname='fraud_db', user='postgres', password='vil100sr', host='localhost', port='5432')
cursor = conn.cursor()

# Function to flatten any sequence values to scalars (fixes TypeError)
def flatten_features(d):
    flat = {}
    for k, v in d.items():
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 1:
            flat[k] = v[0]
        elif isinstance(v, str):
            flat[k] = v  # Keep strings for categorical encoding
        else:
            try:
                flat[k] = float(v)  # Ensure numeric
            except (ValueError, TypeError):
                flat[k] = 0.0  # Fallback for invalid
    return flat

# New: Safe coord validation to fix ValueError (checks range, converts to float, swaps if needed)
def safe_coords(lat, long):
    try:
        lat = float(lat)
        long = float(long)
        # If lat is out of [-90,90], assume swapped with long and fix
        if not (-90 <= lat <= 90):
            lat, long = long, lat  # Swap
            if not (-90 <= lat <= 90):  # Still invalid?
                raise ValueError(f"Invalid latitude after swap: {lat}")
        if not (-180 <= long <= 180):
            raise ValueError(f"Invalid longitude: {long}")
        return (lat, long)
    except Exception as e:
        print(f"Coord error: {e} - Using default (0,0)")
        return (0.0, 0.0)  # Safe fallback to avoid crash

# River online pipeline: TransformerUnion to combine encoded categoricals and scaled numerics + LogisticRegression
# Updated: Added optimizer with higher learning rate for faster adaptation
online_model = compose.Pipeline(
    compose.TransformerUnion(
        compose.Select('type') | preprocessing.OneHotEncoder(),
        compose.Select('amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'step', 'isFlaggedFraud') | preprocessing.StandardScaler()
    ),
    linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.1))  # Increased from default 0.01
)

# Online metric and drift detector
metric = metrics.Accuracy()  # Tracks online accuracy
drift_detector = drift.ADWIN(delta=0.01)  # Updated: Less sensitive (from 0.001) to avoid over-resetting

# Added: Warm-up the model with initial data to reduce initial bias toward 0
warmup_data = [
    ({'type': 'PAYMENT', 'amount': 100.0, 'oldbalanceOrg': 1000.0, 'newbalanceOrig': 900.0, 'oldbalanceDest': 0.0, 'newbalanceDest': 100.0, 'step': 1, 'isFlaggedFraud': 0}, 0),
    ({'type': 'TRANSFER', 'amount': 50000.0, 'oldbalanceOrg': 0.0, 'newbalanceOrig': 0.0, 'oldbalanceDest': 0.0, 'newbalanceDest': 50000.0, 'step': 10, 'isFlaggedFraud': 1}, 1),
    ({'type': 'CASH_IN', 'amount': 500.0, 'oldbalanceOrg': 2000.0, 'newbalanceOrig': 2500.0, 'oldbalanceDest': 100.0, 'newbalanceDest': 0.0, 'step': 2, 'isFlaggedFraud': 0}, 0),
    ({'type': 'CASH_OUT', 'amount': 100000.0, 'oldbalanceOrg': 100.0, 'newbalanceOrig': 0.0, 'oldbalanceDest': 0.0, 'newbalanceDest': 100000.0, 'step': 20, 'isFlaggedFraud': 1}, 1),
    ({'type': 'DEBIT', 'amount': 50.0, 'oldbalanceOrg': 500.0, 'newbalanceOrig': 450.0, 'oldbalanceDest': 0.0, 'newbalanceDest': 50.0, 'step': 3, 'isFlaggedFraud': 0}, 0),
]
for x, y in warmup_data:
    online_model.learn_one(x, y)
print("Model warmed up with initial data.")

# Kafka consumer config
consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'fraud_detector',
    'auto.offset.reset': 'earliest'
})
consumer.subscribe(['fraud_transactions'])

# Process loop
while True:
    msg = consumer.poll(1.0)
    if msg is None:
        continue
    if msg.error():
        print(f"Error: {msg.error()}")
        continue

    # Parse message into dict
    features_dict = json.loads(msg.value().decode('utf-8'))

    # Flatten to ensure scalars
    features_dict = flatten_features(features_dict)

    # Preprocess for offline model
    df = pd.DataFrame([features_dict])
    processed = preprocessor.transform(df)

    # Offline prediction
    offline_pred = offline_model.predict(processed)[0]  # Scalar output

    # Simulate feedback loop
    if offline_pred == 1:
        confirmed_label = 1 if random.random() < 0.8 else 0
    else:
        confirmed_label = 1 if random.random() < 0.2 else 0

    # Incremental update with River
    online_model.learn_one(features_dict, confirmed_label)

    # Online prediction
    online_pred = online_model.predict_one(features_dict)

    # Update metric
    metric.update(confirmed_label, online_pred)

    # Added: Check for concept drift using prediction error
    prediction_error = abs(confirmed_label - online_pred)  # 0 or 1 error
    drift_detector.update(prediction_error)
    if drift_detector.drift_detected:
        print("Concept drift detected! Resetting online model for adaptation.")
        # Reset to fresh model on drift
        online_model = compose.Pipeline(
            compose.TransformerUnion(
                compose.Select('type') | preprocessing.OneHotEncoder(),
                compose.Select('amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'step', 'isFlaggedFraud') | preprocessing.StandardScaler()
            ),
            linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.1))  # Maintain higher lr on reset
        )

    # Added: Geo fraud check (lookup user from DB and calculate distance)
    user_id = features_dict.get('user_id', '')
    try:
        cursor.execute("SELECT billing_lat, billing_long FROM users WHERE user_id = %s", (user_id,))
        user_row = cursor.fetchone()
        if user_row is None:
            print("User not found in DB, skipping geo check")
            geo_flag = 0
            distance_km = 0.0
        else:
            home_lat, home_long = user_row
            txn_lat = features_dict.get('tx_lat', 0.0)
            txn_long = features_dict.get('tx_long', 0.0)
            home_coords = safe_coords(home_lat, home_long)
            txn_coords = safe_coords(txn_lat, txn_long)
            distance_km = geodesic(home_coords, txn_coords).km
            geo_flag = 1 if distance_km > 10000 else 0
    except Exception as e:
        print(f"DB query error: {e} - Skipping geo check and rolling back")
        conn.rollback()
        geo_flag = 0
        distance_km = 0.0

    # Combine with existing preds (e.g., OR logic)
    final_pred = 1 if offline_pred == 1 or online_pred == 1 or geo_flag == 1 else 0

    # Log to transactions table
    try:
        cursor.execute("""
            INSERT INTO transactions (user_id, transaction_data, offline_pred, online_pred, geo_flag, final_pred)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (user_id, json.dumps(features_dict), int(offline_pred), int(online_pred), int(geo_flag), int(final_pred)))
        conn.commit()
        print("Successfully inserted transaction for user:", user_id)
    except Exception as e:
        print(f"DB insert error: {e} - Rolling back")
        conn.rollback()

    # Output (enhanced with geo info)
    print(f"Received: {features_dict} | Offline Pred: {offline_pred} | Online Pred: {online_pred} | "
          f"Geo Flag: {geo_flag} (Distance: {distance_km:.2f}km) | Final Pred: {final_pred} | Confirmed Label: {confirmed_label} | Online Accuracy: {metric.get()}")

    consumer.commit()

# Clean up (at end, if script exits)
cursor.close()
conn.close()
