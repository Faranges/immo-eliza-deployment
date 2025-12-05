import os
import pickle
import pandas as pd

# PatH to model (Render-safe)
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgb_pipeline.pkl")

# Load the model once at import time 
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def make_prediction(input_data: dict):
    """
    Takes a dictionary from the API endpoint, converts it to a DataFrame,
    applies the model pipeline, and returns the prediction.
    """

    # Convert incoming JSON to DataFrame
    df = pd.DataFrame([input_data])

    # Predict using the loaded pipeline
    prediction = model.predict(df)[0]

    return {"prediction": float(prediction)}
