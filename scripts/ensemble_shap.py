# scripts/ensemble_shap.py - Ensemble Model with SHAP

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import classification_report
import shap
import joblib
from preprocess import preprocess_data
import matplotlib.pyplot as plt
import os

# Ensure outputs and models folders exist
os.makedirs('../outputs', exist_ok=True)
os.makedirs('../models', exist_ok=True)

# Load and preprocess subset
df = pd.read_csv('../data/paysim.csv').sample(100000, random_state=42)
X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

# Base models
rf = RandomForestClassifier(n_estimators=50, random_state=42)

# LSTM for sequences (reshape to (samples, 1, features))
X_train_lstm = np.expand_dims(X_train, axis=1)
X_test_lstm = np.expand_dims(X_test, axis=1)
lstm = Sequential([LSTM(50, input_shape=(1, X_train.shape[1])), Dense(1, activation='sigmoid')])
lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm.fit(X_train_lstm, y_train, epochs=3, batch_size=32, verbose=1)

# Custom wrapper for LSTM in StackingClassifier (since it expects sklearn-like estimators)
class LSTMWrapper:
    def __init__(self, model):
        self.model = model
    def fit(self, X, y):
        X_lstm = np.expand_dims(X, axis=1)
        self.model.fit(X_lstm, y)
    def predict(self, X):
        X_lstm = np.expand_dims(X, axis=1)
        return (self.model.predict(X_lstm) > 0.5).astype(int)
    def predict_proba(self, X):
        X_lstm = np.expand_dims(X, axis=1)
        return self.model.predict(X_lstm)

lstm_wrapper = LSTMWrapper(lstm)

# Stacked ensemble
estimators = [('rf', rf), ('lstm', lstm_wrapper)]
ensemble = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier(), cv=3)
ensemble.fit(X_train, y_train)

# Predict and evaluate
y_pred = ensemble.predict(X_test)
print(classification_report(y_test, y_pred))

# SHAP explainability (using RF base for tree-based SHAP)
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig('../outputs/shap_summary.png')
plt.close()
print("SHAP summary plot saved to outputs/shap_summary.png")

# Test on mock data
mock_data = pd.DataFrame({'type': ['TRANSFER'], 'amount': [1000000], 'oldbalanceOrg': [0], 'newbalanceOrig': [0], 
                          'oldbalanceDest': [0], 'newbalanceDest': [0], 'step': [1], 'isFlaggedFraud': [0]})
mock_processed = preprocessor.transform(mock_data)
print("Mock prediction:", ensemble.predict(mock_processed))

# Save ensemble
joblib.dump(ensemble, '../models/ensemble_model.pkl')
print("Ensemble model saved to models/")
