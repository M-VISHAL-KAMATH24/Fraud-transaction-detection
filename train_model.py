import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model():
    """
    This function loads the raw dataset, trains a complete preprocessing and
    classification pipeline, and saves the final, unified model to a single file.
    """
    # --- 1. Load Data ---
    print("Loading dataset 'fraud_dataset.csv'...")
    try:
        # Assumes 'fraud_dataset.csv' is in the same root directory
        df = pd.read_csv('paysim.csv')
        # We use a smaller sample for faster training. You can increase this or use the whole dataset.
        df = df.sample(n=50000, random_state=42)
    except FileNotFoundError:
        print("\nFATAL ERROR: 'fraud_dataset.csv' not found.")
        print("Please place the dataset file in the same directory as this script and run again.")
        return

    # --- 2. Define Features and Target ---
    # These are the columns the model will be trained on and will expect during prediction.
    features = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    target = 'isFraud'

    X = df[features]
    y = df[target]

    categorical_features = ['type']
    numerical_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

    # --- 3. Create the Preprocessing Pipeline ---
    # This defines how to handle different data types.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # --- 4. Define the Machine Learning Model ---
    # We will use a RandomForestClassifier.
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # --- 5. Create the Full, Unified Pipeline ---
    # This is the most critical step. It combines preprocessing and the model into one object.
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # --- 6. Train the Pipeline ---
    print("\nTraining the full pipeline... (This may take a few minutes)")
    full_pipeline.fit(X, y)
    print("Training complete.")

    # --- 7. Save the Final, Unified Model ---
    MODELS_DIR = 'models'
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    FINAL_PIPELINE_PATH = os.path.join(MODELS_DIR, 'final_fraud_model.pkl')
    
    print(f"Saving the final, unified pipeline to: {FINAL_PIPELINE_PATH}")
    joblib.dump(full_pipeline, FINAL_PIPELINE_PATH)

    print("\n--- SCRIPT COMPLETE ---")
    print("A new, correct model file has been created. You can now use this in your application.")

if __name__ == '__main__':
    train_and_save_model()
