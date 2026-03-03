import joblib
import pandas as pd
from src.data_preprocessing import preprocess_input


def load_artifacts():
    model = joblib.load("model/attrition_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    feature_names = joblib.load("model/feature_names.pkl")
    return model, scaler, feature_names


def predict_employee(input_dict):
    """
    Takes employee data as dictionary
    Returns probability and risk level
    """

    model, scaler, feature_names = load_artifacts()

    input_df = pd.DataFrame([input_dict])

    input_processed = preprocess_input(input_df, feature_names)

    input_scaled = scaler.transform(input_processed)

    prob = model.predict_proba(input_scaled)[:, 1][0]

    # Risk band logic
    if prob < 0.3:
        risk = "Low Risk"
    elif prob < 0.6:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    return prob, risk