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

    # request is JSON data
    data = request.get_json(force=True)

    # training data columns
    expected_cols = ["cap-diameter", "cap-shape", "cap-surface", "cap-color", "does-bruise-or-bleed", "gill-attachment",
                        "gill-spacing", "gill-color", "stem-height", "stem-width", "stem-root", "stem-surface", "stem-color",
                        "veil-color", "has-ring", "ring-type", "spore-print-color", "habitat", "season"]

    input_df = pd.DataFrame([data])
    input_df = input_df.reindex(columns=expected_cols, fill_value=None) # reorder

    categorical_cols = feature_encoders.keys()
    encoded_features = []
    
    for col in categorical_cols:
        if col in input_df.columns:
            encoded_value = feature_encoders[col].transform([str(input_df.loc[0, col])])
            encoded_features.append(encoded_value[0])

    numerical_cols = ["cap-diameter", "stem-height", "stem-width"]
    numerical_features = []
    for col in numerical_cols:
        if col in input_df.columns:
            try:
                numerical_features.append(float(input_df.loc[0, col]))
            except (ValueError, TypeError):
                numerical_features.append(0.0)

    combined_features = np.array(numerical_features + encoded_features).reshape(1, -1)
    
    input_scaled = scaler.transform(combined_features)
    prediction = model.predict(input_scaled)
    predicted_class = (prediction > 0.5).astype("int32")[0][0] # 0.5 threshold for positive
    predicted_label = target_encoder.inverse_transform([predicted_class])[0]

    return jsonify({
        "prediction": "poisonous" if predicted_label == "p" else "edible",
        "confidence": float(prediction[0][0])
    })

if __name__ == "__main__":
    app.run(debug=True)
