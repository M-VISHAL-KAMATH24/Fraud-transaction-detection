from confluent_kafka import Consumer
import json
import pandas as pd
import joblib
import numpy as np
import random  # For simulating feedback
import os  # To suppress TensorFlow warnings

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# River imports for online learning, preprocessing, and drift detection
from river import compose  # For Pipeline and TransformerUnion
from river import linear_model  # LogisticRegression
from river import metrics  # Accuracy tracking
from river import drift  # ADWIN for concept drift
from river import preprocessing  # For OneHotEncoder and StandardScaler
from river import anomaly  # For HalfSpaceTrees if used

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

# River online pipeline: TransformerUnion to combine encoded categoricals and scaled numerics + LogisticRegression
online_model = compose.Pipeline(
    compose.TransformerUnion(
        compose.Select('type') | preprocessing.OneHotEncoder(),
        compose.Select('amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'step', 'isFlaggedFraud') | preprocessing.StandardScaler()
    ),
    linear_model.LogisticRegression()
)

# Online metric and drift detector
metric = metrics.Accuracy()  # Tracks online accuracy
drift_detector = drift.ADWIN(delta=0.001)  # Added: Detects concept drift; adjust delta for sensitivity

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
    offline_pred = offline_model.predict(processed)  # Scalar output

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
            linear_model.LogisticRegression()
        )

    # Output
    print(f"Received: {features_dict} | Offline Pred: {offline_pred} | Online Pred: {online_pred} | Confirmed Label: {confirmed_label} | Online Accuracy: {metric.get()}")

    consumer.commit()
