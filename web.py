import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "models/")
BEST_MODEL_NAME = os.getenv("BEST_MODEL_NAME", "best_model.pkl")
ONE_HOT_ENCODER_NAME = os.getenv("ONE_HOT_ENCODER_NAME", "one_hot_encoder.pkl")
SCALER_NAME = os.getenv("SCALER_NAME", "scaler.pkl")

CATEGORICAL_FEATURES = ["gender", "who_bmi", "depression_severity", "anxiousness", "depressiveness", "sleepiness"]
NUMERIC_FEATURES = ["age", "bmi", "phq_score", "epworth_score"]

app = Flask(__name__)

model = None
one_hot_encoder = None
scaler = None
config = None

def load_model():
    global model, one_hot_encoder, scaler, config

    try:
        with open(f"{MODEL_DIR}/{BEST_MODEL_NAME}", "rb") as f:
            model = pickle.load(f)

        with open(f"{MODEL_DIR}/{ONE_HOT_ENCODER_NAME}", "rb") as f:
            one_hot_encoder = pickle.load(f)

        with open(f"{MODEL_DIR}/{SCALER_NAME}", "rb") as f:
            scaler = pickle.load(f)

        with open(f"{MODEL_DIR}/config.pkl", "rb") as f:
            config = pickle.load(f)

        print(f"Threshold loaded: {config['threshold']}")

        print("Model and preprocessing objects loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        categorical_data = {feature: data.get(feature, '') for feature in CATEGORICAL_FEATURES}
        numeric_data = {feature: float(data.get(feature, 0)) for feature in NUMERIC_FEATURES}

        print("Received data:", data)
        print("Categorical:", categorical_data)

        categorical_df = pd.DataFrame([categorical_data])
        numeric_df = pd.DataFrame([numeric_data])

        categorical_encoded = one_hot_encoder.transform(categorical_df)
        numeric_scaled = scaler.transform(numeric_df)

        features = np.concatenate([numeric_scaled, categorical_encoded], axis=1)

        probability = model.predict_proba(features)[0]

        print(model.predict_proba(features))

        prob_anxiety = float(probability[1])
        threshold = config['threshold']
        low = config['low_boundary']
        high = config['high_boundary']

        prediction = 1 if prob_anxiety >= threshold else 0

        if prediction == 0:
            risk_level = "Low"
        elif prob_anxiety < low:
            risk_level = "Low"
        elif prob_anxiety < high:
            risk_level = "Medium"
        else:
            risk_level = "High"

        result = {
            'prediction': prediction,
            'probability': prob_anxiety,
            'prediction_label': 'Anxiety Detected' if prediction == 1 else 'No Anxiety',
            'confidence': f"{prob_anxiety * 100:.2f}%",
            'risk_level': risk_level
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    if load_model():
        app.run(debug=True, host='0.0.0.0', port=8080)
    else:
        print("Failed to load model. Please run train.py first to create the model.")
