import pandas as pd
import joblib
import os

def load_artifacts():
    """Loads the trained model, scaler, and expected feature names."""
    # We will use your Random Forest model as the primary predictor
    model = joblib.load(os.path.join("model", "random_forest_model.pkl"))
    scaler = joblib.load(os.path.join("model", "scaler.pkl"))
    feature_names = joblib.load(os.path.join("model", "feature_names.pkl"))
    
    return model, scaler, feature_names

def predict_employee_attrition(employee_data_dict):
    """
    Takes a dictionary of employee data from the UI, processes it, 
    and returns an attrition prediction and probability.
    """
    model, scaler, feature_names = load_artifacts()
    
    # 1. Convert the single employee's dictionary into a Pandas DataFrame
    df = pd.DataFrame([employee_data_dict])
    
    # 2. One-hot encode the categorical variables (like Department, Job Role)
    df_encoded = pd.get_dummies(df)
    
    # 3. THE MAGIC TRICK: Align the UI columns with the Training columns
    # The model expects exactly the same columns it was trained on. 
    # This reindex command adds missing columns (filling them with 0) 
    # and ensures they are in the exact right order.
    df_aligned = df_encoded.reindex(columns=feature_names, fill_value=0)
    
    # 4. Scale the numerical features using the saved scaler
    X_scaled = scaler.transform(df_aligned)
    
    # 5. Make the prediction
    prediction = model.predict(X_scaled)[0] # Returns 0 (No) or 1 (Yes)
    probability = model.predict_proba(X_scaled)[0][1] # Probability of leaving
    
    return prediction, probability

# Small test block to make sure the file works if run directly
if __name__ == "__main__":
    # A dummy employee to test the script
    test_employee = {
        "Age": 30,
        "Department": "Sales",
        "DistanceFromHome": 10,
        "JobRole": "Sales Executive",
        "MonthlyIncome": 5000,
        "OverTime": "Yes",
        "YearsAtCompany": 3,
        "JobSatisfaction": 2,
        "PerformanceRating": 3
    }
    
    try:
        pred, prob = predict_employee_attrition(test_employee)
        print(f"Test Successful! Prediction: {pred}, Probability: {prob:.2f}")
    except Exception as e:
        print(f"Error during testing: {e}")