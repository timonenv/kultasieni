"""
Code for API to run the model. Requires the trained model from model.py.
"""

import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from tensorflow import keras
import numpy as np

MODEL_FILENAME = "kultasieni_classifier.keras"
SCALER_FILENAME = "kultasieni_scaler.joblib"
FEATURE_ENCODERS_FILENAME = "kultasieni_feature_encoders.joblib"
TARGET_ENCODER_FILENAME = "kultasieni_target_encoder.joblib"

# must have access to the below files
try:
    model = keras.models.load_model(MODEL_FILENAME) # .keras filename
    scaler = joblib.load(SCALER_FILENAME) # .joblib filename
    feature_encoders = joblib.load(FEATURE_ENCODERS_FILENAME)
    target_encoder = joblib.load(TARGET_ENCODER_FILENAME)
except Exception as e:
    print("Error loading model or preprocessing files: {}".format(e))
    model = scaler = feature_encoders = target_encoder = None

app = Flask(__name__)
CORS(app) # if not working with github
@app.route("/predict", methods=["POST", "OPTIONS"])
@cross_origin()
def predict():
    """
    Create prediction from model: poisonous/edible.
    """
    if request.method == "OPTIONS":
        return "", 200

    if model is None or scaler is None or feature_encoders is None or target_encoder is None:
        return jsonify({"error": "Model, scaler, or encoder not loaded. Please train the model first."}), 500
    
    try:    
        # request is JSON data
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON data received."}), 400

        expected_cols = ["cap-diameter", "cap-shape", "cap-surface", "cap-color", "does-bruise-bleed", "gill-attachment",
                            "gill-spacing", "gill-color", "stem-height", "stem-width", "stem-root", "stem-surface", "stem-color",
                            "veil-color", "has-ring", "ring-type", "spore-print-color", "habitat", "season"]

        input_df = pd.DataFrame([data])
        input_df = input_df.reindex(columns=expected_cols, fill_value=None)
        
        processed_features = []
        
        for col in expected_cols:
            value = input_df.loc[0, col]
            if col in ["cap-diameter", "stem-height", "stem-width"]:
                try:
                    processed_features.append(float(value) if pd.notna(value) else 0.0)
                except (ValueError, TypeError):
                    processed_features.append(0.0) 
            else:
                try:
                    encoder = feature_encoders.get(col)
                    if encoder:
                        encoded_value = encoder.transform([str(value)])
                        processed_features.append(encoded_value[0])
                    else:
                        processed_features.append(0)
                except ValueError as e:
                    return jsonify({"error": f"Unknown label for column '{col}': {e}"}), 400
                except (ValueError, TypeError):
                    processed_features.append(0) 
        
        combined_df = pd.DataFrame([processed_features], columns=expected_cols)
        
        # scale and predict
        input_scaled = scaler.transform(combined_df)
        prediction = model.predict(input_scaled)
        predicted_class = (prediction > 0.5).astype("int32")[0][0] # 0.5 threshold for positive
        predicted_label = target_encoder.inverse_transform([predicted_class])[0]

        return jsonify({
            "prediction": "poisonous" if predicted_label == "p" else "edible",
            "confidence": float(prediction[0][0])
        })
    
    except Exception as e:
        print(e)
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
