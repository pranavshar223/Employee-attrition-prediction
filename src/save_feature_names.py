import pandas as pd
import joblib
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#  Correct model folder path
model_dir = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

#  Dataset load
data_path = os.path.join(BASE_DIR, "data", "Processed", "processed_hr_attrition.csv")
df = pd.read_csv(data_path)

#  Same preprocessing as training
X = df.drop("Attrition", axis=1)
X = pd.get_dummies(X, drop_first=True)

#  Save feature names
feature_path = os.path.join(model_dir, "feature_names.pkl")
joblib.dump(X.columns.tolist(), feature_path)

print("feature_names.pkl saved successfully ✅")
print("Saved at:", feature_path)
print("Files in model folder:", os.listdir(model_dir))