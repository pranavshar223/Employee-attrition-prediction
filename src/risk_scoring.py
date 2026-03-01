import pandas as pd
import numpy as np
import joblib
import os

os.makedirs("data/output", exist_ok=True)

model = joblib.load("../model/attrition_model.pkl")
scaler = joblib.load("../model/scaler.pkl")
feature_names = joblib.load("../model/feature_names.pkl")

df = pd.read_csv("data/processed/processed_hr_attrition.csv")

new_employees = df.drop("Attrition",axis=1).copy()
new_employees = pd.get_dummies(new_employees ,drop_first=True)
new_employees = new_employees.reindex(columns=feature_names, fill_value=0)
new_employees_scaled = scaler.transform(new_employees)

risk_prob = model.predict_proba(new_employees_scaled)[:, 1]

def get_risk_level(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"
    
risk_level = [get_risk_level(p) for p in risk_prob] 

result  = new_employees.copy()

result["Attrition_Probability"] = risk_prob
result["Risk_Level"] = risk_level

print(result[["Attrition_Probability", "Risk_Level"]].head())

result.to_csv("data/output/attrition_risk_output.csv", index=False)