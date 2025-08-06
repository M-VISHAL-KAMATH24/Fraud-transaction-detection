# scripts/basic_training.py - Basic RandomForest Training and Evaluation

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import joblib
from preprocess import preprocess_data  # Import from same folder
import os

# Ensure models folder exists
os.makedirs('../models', exist_ok=True)

# Load and preprocess (use subset for speed, e.g., 100k rows)
df = pd.read_csv('../data/paysim.csv').sample(100000, random_state=42)
X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

# Train RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))  # Aim for 80%+ recall
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and preprocessor
joblib.dump(model, '../models/basic_rf_model.pkl')
joblib.dump(preprocessor, '../models/preprocessor.pkl')
print("Model and preprocessor saved to models/")
