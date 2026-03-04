import streamlit as st
import pandas as pd
import sys
import os

# Ensure Python can find the 'src' folder to import your predict script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict_employee_attrition

# 1. Page Configuration
st.set_page_config(page_title="HR Attrition Predictor", page_icon="🏢", layout="wide")

def main():
    st.title("Employee Attrition Prediction & HR Insights")
    st.markdown("Predict the likelihood of an employee leaving the company and analyze key retention factors.")

    # 2. Sidebar Navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a module:", 
                                ["Single Employee Prediction", "Batch Analysis", "Dashboard"])

    # 3. Module: Single Employee Prediction
    if app_mode == "Single Employee Prediction":
        st.header("Predict Attrition Risk for a Single Employee")
        
        # 3-column layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=65, value=30)
            department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            distance = st.number_input("Distance From Home (miles)", min_value=1, max_value=50, value=5)
            
        with col2:
            job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manager", "Healthcare Representative"])
            # Added Salary_per_hour because your model found it was the #1 most important feature!
            salary_per_hour = st.number_input("Salary Per Hour ($)", min_value=10, max_value=500, value=50) 
            overtime = st.selectbox("OverTime", ["Yes", "No"])
            
        with col3:
            years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=3)
            # Added TotalWorkingYears because it was the #3 most important feature
            total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=5) 
            job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)

        st.markdown("---")
        
        # The Prediction Button
        if st.button("Calculate Risk Score", type="primary"):
            
            # 1. Gather the UI inputs into a dictionary
            employee_data = {
                "Age": age,
                "Department": department,
                "DistanceFromHome": distance,
                "JobRole": job_role,
                "Salary_per_hour": salary_per_hour,
                "OverTime": overtime,
                "YearsAtCompany": years_at_company,
                "TotalWorkingYears": total_working_years,
                "JobSatisfaction": job_satisfaction,
                # Setting baseline defaults for a few other features the model expects
                "DailyRate": 800,
                "HourlyRate": 65,
                "YearsWithCurrManager": 2,
                "StockOptionLevel": 0,
                "NumCompaniesWorked": 1
            }
            
            # 2. Call the prediction script
            try:
                prediction, probability = predict_employee_attrition(employee_data)
                
                # 3. Display the Output
                st.subheader("Prediction Results")
                if prediction == 1:
                    st.error("⚠️ **High Flight Risk!**")
                    st.write(f"The model predicts a **{probability*100:.1f}%** chance this employee will leave the company.")
                else:
                    st.success("✅ **Low Flight Risk**")
                    st.write(f"The model predicts a **{probability*100:.1f}%** chance this employee will leave. They are likely to stay.")
                    
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    # 4. Module: Batch Analysis
    elif app_mode == "Batch Analysis":
        st.header("Batch Prediction via CSV Upload")
        uploaded_file = st.file_uploader("Upload HR Data", type=["csv"])
        if uploaded_file is not None:
            df_preview = pd.read_csv(uploaded_file)
            st.dataframe(df_preview.head())

    # 5. Module: Dashboard
    elif app_mode == "Dashboard":
        st.header("HR Insights Dashboard")
        st.info("Visualizations will appear here.")

if __name__ == "__main__":
    main()