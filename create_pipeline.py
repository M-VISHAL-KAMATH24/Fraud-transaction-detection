import joblib
from sklearn.pipeline import Pipeline
import os
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# ======================================================================
# === THE DEFINITIVE FIX IS TO ADD THIS CLASS DEFINITION ===
# ======================================================================
# This class definition must be present for joblib to load the old model.
class LSTMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None): # y can be None for pre-trained models
        return self

    def predict(self, X):
        X_lstm = np.expand_dims(X, axis=1) if len(X.shape) == 1 else X
        return (self.model.predict(X_lstm) > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        X_lstm = np.expand_dims(X, axis=1) if len(X.shape) == 1 else X
        proba_class_1 = self.model.predict(X_lstm)
        proba_class_0 = 1 - proba_class_1
        return np.hstack((proba_class_0, proba_class_1))

    def get_params(self, deep=True):
        return {"model": self.model}
# ======================================================================

# --- Define Paths (unchanged) ---
MODELS_DIR = 'models'
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, 'preprocessor.pkl')
MODEL_PATH = os.path.join(MODELS_DIR, 'ensemble_model.pkl')
FINAL_PIPELINE_PATH = os.path.join(MODELS_DIR, 'full_fraud_pipeline.pkl')

def create_and_save_pipeline():
    """
    Loads the separate model files and combines them into a single,
    unified scikit-learn pipeline.
    """
    print("--- Starting Model Repair Process ---")
    
    try:
        print(f"Loading preprocessor from: {PREPROCESSOR_PATH}")
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("Preprocessor loaded successfully.")
        
        print(f"Loading ensemble model from: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        print("Ensemble model loaded successfully.")

    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not find model files in '{MODELS_DIR}'. Details: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading model files: {e}")
        return

    # Create the single, unified pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    print("\nSuccessfully created a unified pipeline object.")

    # Save the new, correct pipeline to a single file
    joblib.dump(full_pipeline, FINAL_PIPELINE_PATH)
    
    print(f"\n--- Model Repair Complete! ---")
    print(f"A new, correct model file has been saved to: {FINAL_PIPELINE_PATH}")

if __name__ == '__main__':
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created directory: {MODELS_DIR}. Please place model files inside.")
    else:
        create_and_save_pipeline()

