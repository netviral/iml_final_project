import sys
import json
import warnings
from helper_functions import process_text
import joblib
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# Load model and feature columns
model = joblib.load("model/saved_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

def predict_text(text):
    metrics = process_text(text)
    metrics_values = [metrics[col] for col in feature_columns]
    prediction_probabilities = model.predict_proba([metrics_values])[0]
    prediction = model.predict([metrics_values])[0]
    return {
        "prediction": "AI" if prediction == 1 else "Human",
        "probability_AI_generated": round(prediction_probabilities[1], 2),
        "probability_human_written": round(prediction_probabilities[0], 2),
    }

# Continuously read from stdin
while True:
    try:
        line = sys.stdin.readline()
        if not line:
            break  # Exit if no input
        data = json.loads(line.strip())  # Expecting JSON input
        text = data.get("text", "")
        if not text:
            raise ValueError("No text provided")
        result = predict_text(text)
        print(json.dumps(result), flush=True)  # Output JSON
    except Exception as e:
        print(json.dumps({"error": str(e)}), flush=True)
