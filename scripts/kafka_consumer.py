from confluent_kafka import Consumer
import json
import pandas as pd
import joblib
import numpy as np  # For reshaping if needed

# Define LSTMWrapper class (copied from ensemble_shap.py for unpickling)
from sklearn.base import BaseEstimator, ClassifierMixin

class LSTMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
        self.classes_ = None  # Initialize classes_ attribute

    def fit(self, X, y):
        self.classes_ = np.unique(y)  # Set classes_ to unique labels in y
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

# Load Week 1 models (now with LSTMWrapper defined)
model = joblib.load('models/ensemble_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# Kafka config
consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'fraud_detector',
    'auto.offset.reset': 'earliest'  # Start from beginning if no offset
})
consumer.subscribe(['fraud_transactions'])

# Process loop (runs forever; Ctrl+C to stop)
while True:
    msg = consumer.poll(1.0)  # Wait 1s for message
    if msg is None:
        continue
    if msg.error():
        print(f"Error: {msg.error()}")
        continue

    # Parse and preprocess
    data = json.loads(msg.value().decode('utf-8'))
    df = pd.DataFrame([data])
    processed = preprocessor.transform(df)

    # Predict (handle if model expects specific shape)
    prediction = model.predict(processed)[0]
    print(f"Received: {data} | Predicted Fraud: {prediction} (1 = Fraud)")

    # Commit to avoid reprocessing
    consumer.commit()
