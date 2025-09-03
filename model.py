"""
Code to create a Keras model for classifying mushroom data on a binary scale (poisonous or edible).
Data from: https://www.kaggle.com/datasets/dhinaharp/mushroom-dataset/data

expected_cols = ["cap-diameter", "cap-shape", "cap-surface", "cap-color", "does-bruise-or-bleed", "gill-attachment",
                    "gill-spacing", "gill-color", "stem-height", "stem-width", "stem-root", "stem-surface", "stem-color",
                    "veil-color", "has-ring", "ring-type", "spore-print-color", "habitat", "season"]
#large keras model 10 epoch:
Model Accuracy: 0.99
Model Loss: 0.0300

# random forest Model Accuracy: 1.00
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
from tensorflow import keras
import os

warnings.filterwarnings("ignore")

USE_KERAS_MODEL = False

KULTASIENI_DATASET_PATH = "secondary_data.csv" # from Kaggle
if USE_KERAS_MODEL:
    MODEL_FILENAME = "kultasieni_classifier.keras"
else:
    MODEL_FILENAME = "kultasieni_classifier.joblib"
SCALER_FILENAME = "kultasieni_scaler.joblib"
FEATURE_ENCODERS_FILENAME = "kultasieni_feature_encoders.joblib"
TARGET_ENCODER_FILENAME = "kultasieni_target_encoder.joblib"

def prepare_data(df):
    """
    Prepares the dataset for classification.
    """
    print("Preparing the dataset.")
    df_processed = df.copy()

    # target label
    y = df_processed["class"]
    X = df_processed.drop("class", axis=1)

    # get rid of bad columns e.g. veil_type with only 1 value
    X.replace("?", pd.NA, inplace=True)

    cols_to_drop = []
    for col in X.columns:
        if X[col].nunique(dropna=True) <= 1:
            cols_to_drop.append(col)
    
    if cols_to_drop:
        print("Dropping columns with low variance: {}".format(cols_to_drop))
        X.drop(columns=cols_to_drop, inplace=True)

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    feature_encoders = {}
    
    # labelencoder converts categories to numerical
    # must be a separate fitted item for each column!
    print("Applying LabelEncoder to categorical columns: {}".format(list(categorical_cols)))
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0]) # impute with mode
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col].astype(str))
        feature_encoders[col] = encoder

    # save fitted encoders
    print("Saving the feature encoders to {}".format(FEATURE_ENCODERS_FILENAME))
    joblib.dump(feature_encoders, FEATURE_ENCODERS_FILENAME)

    print("Processing numerical columns: {}".format(list(numerical_cols)))
    for col in numerical_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].mean()) # impute with mean

    print("Scaling features using MinMaxScaler.")
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    print("Saving the scaler to {}".format(SCALER_FILENAME))
    joblib.dump(scaler, SCALER_FILENAME)

    print("Dataset prepared successfully.")
    return X, y

def train_keras_model(X_train, y_train):
    """
    Trains the model.
    """
    print("Training the Keras deep learning model...")
    
    target_encoder = LabelEncoder()
    y_train_encoded = target_encoder.fit_transform(y_train) # change target to binary 0 and 1
    print("Saving the target encoder to {}".format(TARGET_ENCODER_FILENAME))
    joblib.dump(target_encoder, TARGET_ENCODER_FILENAME)

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    model.fit(X_train, y_train_encoded, epochs=50, batch_size=32, verbose=1)
    print("Model training complete.")
    return model

def evaluate_keras_model(model, X_test, y_test):
    """
    Model performance evaluation.
    """
    print("Evaluating the model...")
    target_encoder = joblib.load(TARGET_ENCODER_FILENAME)
    y_test_encoded = target_encoder.transform(y_test)
    loss, accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Model Loss: {loss:.4f}")

def save_keras_model(model):
    """
    Save the trained Keras model to a file.
    """
    print("Saving the model to {}".format(MODEL_FILENAME))
    model.save(MODEL_FILENAME)
    print("Model saved successfully.")

def train_scikit_model(X_train, y_train):
    """
    Trains the scikit-learn Naive Bayes model.
    """
    print("Training the scikit-learn Random Forest model...")
    
    target_encoder = LabelEncoder()
    y_train_encoded = target_encoder.fit_transform(y_train)
    print("Saving the target encoder to {}".format(TARGET_ENCODER_FILENAME))
    joblib.dump(target_encoder, TARGET_ENCODER_FILENAME)

    # Use the more powerful RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train_encoded)
    
    print("Model training complete.")
    return model

def evaluate_scikit_model(model, X_test, y_test):
    print("Evaluating the model...")
    target_encoder = joblib.load(TARGET_ENCODER_FILENAME)
    y_test_encoded = target_encoder.transform(y_test)
    accuracy = model.score(X_test, y_test_encoded)
    print("Model Accuracy: {:.2f}".format(accuracy))

def save_scikit_model(model):
    print("Saving the scikit-learn model to {}".format(MODEL_FILENAME))
    joblib.dump(model, MODEL_FILENAME)
    print("Model saved successfully.")


def main():
    df = pd.read_csv(KULTASIENI_DATASET_PATH, sep=";")
    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split: {} training, {} testing.".format(len(X_train), len(X_test)))

    if USE_KERAS_MODEL:
        print("Using DL Keras model")
        model = train_keras_model(X_train, y_train)
        evaluate_keras_model(model, X_test, y_test)
        save_keras_model(model)
    else:
        print("Using lightweight scikit model")
        model = train_scikit_model(X_train, y_train)
        evaluate_scikit_model(model, X_test, y_test)
        save_scikit_model(model)

if __name__ == "__main__":
    main()