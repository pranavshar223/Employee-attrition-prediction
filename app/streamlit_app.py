import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

from src.predict import predict_employee
from src.recommendations import generate_recommendations
from src.generate_report import generate_employee_report


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

# --------------------------------------------------
# Load Random Forest Feature Importance
# --------------------------------------------------

@st.cache_resource
def load_feature_importance():

    model = joblib.load("model/attrition_model.pkl")
    feature_names = joblib.load("model/feature_names.pkl")

    importance = model.coef_[0]

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    feature_importance["AbsoluteImportance"] = (
        feature_importance["Importance"].abs()
    )

    feature_importance = feature_importance.sort_values(
        by="AbsoluteImportance",
        ascending=False
    )

    return feature_importance

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("Employee Attrition Prediction")
st.write(
    "Predict the probability of employee attrition using Machine Learning."
)

st.divider()


# --------------------------------------------------
# Employee Information
# --------------------------------------------------

st.header("Employee Information")
st.write("Enter the employee details below to predict attrition risk.")


# --------------------------------------------------
# Personal Information
# --------------------------------------------------

st.subheader("👤 Personal Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=30
    )

with col2:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col3:
    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )


# --------------------------------------------------
# Job Information
# --------------------------------------------------

st.subheader("💼 Job Information")

col1, col2, col3 = st.columns(3)

with col1:
    department = st.selectbox(
        "Department",
        [
            "Sales",
            "Research & Development",
            "Human Resources"
        ]
    )

with col2:
    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )

with col3:
    business_travel = st.selectbox(
        "Business Travel",
        [
            "Travel_Rarely",
            "Travel_Frequently",
            "Non-Travel"
        ]
    )


# --------------------------------------------------
# Education Information
# --------------------------------------------------

st.subheader("🎓 Education")

col1, col2 = st.columns(2)

with col1:
    education = st.number_input(
        "Education Level",
        min_value=1,
        max_value=5,
        value=3
    )

with col2:
    education_field = st.selectbox(
        "Education Field",
        [
            "Life Sciences",
            "Medical",
            "Marketing",
            "Technical Degree",
            "Other",
            "Human Resources"
        ]
    )


# --------------------------------------------------
# Compensation
# --------------------------------------------------

st.subheader("💰 Compensation")

col1, col2, col3 = st.columns(3)

with col1:
    daily_rate = st.number_input(
        "Daily Rate",
        min_value=0,
        value=800
    )

with col2:
    hourly_rate = st.number_input(
        "Hourly Rate",
        min_value=0,
        value=65
    )

with col3:
    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0,
        value=5000
    )


# --------------------------------------------------
# Job Satisfaction
# --------------------------------------------------

st.subheader("❤️ Employee Satisfaction")

col1, col2, col3 = st.columns(3)

with col1:
    environment_satisfaction = st.slider(
        "Environment Satisfaction",
        min_value=1,
        max_value=4,
        value=3
    )

with col2:
    job_satisfaction = st.slider(
        "Job Satisfaction",
        min_value=1,
        max_value=4,
        value=3
    )

with col3:
    relationship_satisfaction = st.slider(
        "Relationship Satisfaction",
        min_value=1,
        max_value=4,
        value=3
    )


# --------------------------------------------------
# Work Information
# --------------------------------------------------

st.subheader("🏢 Work Information")

col1, col2, col3 = st.columns(3)

with col1:
    job_level = st.number_input(
        "Job Level",
        min_value=1,
        max_value=5,
        value=2
    )

with col2:
    job_involvement = st.slider(
        "Job Involvement",
        min_value=1,
        max_value=4,
        value=3
    )

with col3:
    total_working_years = st.number_input(
        "Total Working Years",
        min_value=0,
        max_value=50,
        value=8
    )


col1, col2, col3 = st.columns(3)

with col1:
    years_at_company = st.number_input(
        "Years at Company",
        min_value=0,
        max_value=50,
        value=5
    )

with col2:
    years_current_role = st.number_input(
        "Years in Current Role",
        min_value=0,
        max_value=20,
        value=3
    )

with col3:
    years_since_promotion = st.number_input(
        "Years Since Last Promotion",
        min_value=0,
        max_value=20,
        value=1
    )


col1, col2, col3 = st.columns(3)

with col1:
    years_current_manager = st.number_input(
        "Years With Current Manager",
        min_value=0,
        max_value=20,
        value=3
    )

with col2:
    num_companies_worked = st.number_input(
        "Number of Companies Worked",
        min_value=0,
        max_value=20,
        value=2
    )

with col3:
    training_times = st.number_input(
        "Training Times Last Year",
        min_value=0,
        max_value=10,
        value=3
    )


# --------------------------------------------------
# Other Factors
# --------------------------------------------------

st.subheader("📊 Other Factors")

col1, col2, col3 = st.columns(3)

with col1:
    distance_from_home = st.number_input(
        "Distance From Home",
        min_value=0,
        value=5
    )

with col2:
    percent_salary_hike = st.number_input(
        "Percent Salary Hike",
        min_value=0,
        max_value=100,
        value=15
    )

with col3:
    stock_option_level = st.number_input(
        "Stock Option Level",
        min_value=0,
        max_value=3,
        value=1
    )


col1, col2, col3 = st.columns(3)

with col1:
    performance_rating = st.slider(
        "Performance Rating",
        min_value=1,
        max_value=5,
        value=3
    )

with col2:
    work_life_balance = st.slider(
        "Work Life Balance",
        min_value=1,
        max_value=4,
        value=3
    )

with col3:
    overtime = st.selectbox(
        "OverTime",
        ["Yes", "No"]
    )


# --------------------------------------------------
# Predict Button
# --------------------------------------------------

st.divider()

predict_button = st.button(
    "🔮 Predict Attrition Risk",
    type="primary",
    use_container_width=True
)

# --------------------------------------------------
# Load Dashboard Data
# --------------------------------------------------

@st.cache_data
def load_dashboard_data():

    risk_path = Path("data/Processed/attrition_risk_output.csv")
    processed_path = Path("data/Processed/processed_hr_attrition.csv")

    risk_df = pd.read_csv(risk_path)
    processed_df = pd.read_csv(processed_path)

    return risk_df, processed_df


# --------------------------------------------------
# Prediction
# --------------------------------------------------
if predict_button:

    # --------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------

    daily_hours = 10 if overtime == "Yes" else 8

    monthly_working_hours = daily_hours * 22

    salary_per_hour = monthly_income / monthly_working_hours


    # --------------------------------------------------
    # Employee Data
    # --------------------------------------------------

    employee_data = {
        "Age": age,
        "DailyRate": daily_rate,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EnvironmentSatisfaction": environment_satisfaction,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobSatisfaction": job_satisfaction,
        "NumCompaniesWorked": num_companies_worked,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_current_role,
        "YearsSinceLastPromotion": years_since_promotion,
        "YearsWithCurrManager": years_current_manager,

        # Categorical features
        "Gender": gender,
        "BusinessTravel": business_travel,
        "Department": department,
        "EducationField": education_field,
        "JobRole": job_role,
        "MaritalStatus": marital_status,
        "OverTime": overtime,

        # Engineered features
        "DailyHours": daily_hours,
        "MonthlyWorkingHours": monthly_working_hours,
        "Salary_per_hour": salary_per_hour
    }


    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    try:

        probability, risk = predict_employee(employee_data)

        st.divider()

        st.header("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Attrition Probability",
                f"{probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Risk Level",
                risk
            )

        st.subheader("👤 Employee Profile")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.write(f"**Age:** {age}")

        with col2:
            st.write(f"**Gender:** {gender}")

        with col3:
            st.write(f"**Department:** {department}")

        with col4:
            st.write(f"**Job Role:** {job_role}")
        
        # --------------------------------------------------
        # Probability
        # --------------------------------------------------

        st.subheader("Attrition Probability")

        st.progress(
            min(max(probability, 0.0), 1.0)
        )

        st.write(
            f"The model estimates a **{probability * 100:.2f}%** "
            "probability of employee attrition."
        )


        # --------------------------------------------------
        # Risk Explanation
        # --------------------------------------------------

        if risk == "High Risk":

            st.error(
                "🚨 High Attrition Risk\n\n"
                "This employee has a high predicted probability "
                "of attrition. HR intervention should be considered."
            )

        elif risk == "Medium Risk":

            st.warning(
                "⚠️ Medium Attrition Risk\n\n"
                "This employee has a moderate predicted probability "
                "of attrition. The employee may benefit from further "
                "HR assessment."
            )

        else:

            st.success(
                "✅ Low Attrition Risk\n\n"
                "This employee currently has a low predicted "
                "probability of attrition."
            )

        # --------------------------------------------------
        # HR Recommendations
        # --------------------------------------------------

        st.divider()

        st.subheader("💡 HR Recommendations")

        st.write(
            "The following recommendations are based on the "
            "employee's current attributes and predicted risk level."
        )

        recommendations = generate_recommendations(
            employee_data,
            risk
        )

        for recommendation in recommendations:
            st.markdown(
                f"• {recommendation}"
            )

        st.divider()
        st.subheader("📄 Employee Risk Report")

        pdf_report = generate_employee_report(
            employee_data,
            probability,
            risk,
            recommendations
        )

        st.download_button(
            label="📥 Download Employee Risk Report",
            data=pdf_report,
            file_name="employee_attrition_risk_report.pdf",
            mime="application/pdf"
        )


        # --------------------------------------------------
        # Key Model Factors
        # --------------------------------------------------

        st.divider()

        st.subheader("🔍 Key Model Factors")

        st.write(
            "These are the features with the strongest influence "
            "on the Logistic Regression model."
        )

        try:

            feature_importance = load_feature_importance()

            top_features = feature_importance.head(10).copy()

            top_features["Feature"] = (
                top_features["Feature"]
                .str.replace("_", " ", regex=False)
            )

            top_features = top_features.sort_values(
                by="Importance",
                ascending=True
            )

            st.bar_chart(
                top_features.set_index("Feature")["Importance"]
            )

        except Exception as e:

            st.warning(
                f"Could not load feature importance: {e}"
            )


    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )

# ==================================================
# HR DASHBOARD
# ==================================================

st.divider()

st.header("📊 HR Attrition Dashboard")

st.write(
    "Overview of employee attrition risk based on the "
    "trained machine learning model."
)


# --------------------------------------------------
# Load Data
# --------------------------------------------------

try:

    risk_df, processed_df = load_dashboard_data()


    # --------------------------------------------------
    # Summary Metrics
    # --------------------------------------------------

    total_employees = len(risk_df)

    high_risk = (
        risk_df["Risk_Level"] == "High Risk"
    ).sum()

    medium_risk = (
        risk_df["Risk_Level"] == "Medium Risk"
    ).sum()

    low_risk = (
        risk_df["Risk_Level"] == "Low Risk"
    ).sum()

    average_probability = (
        risk_df["Attrition_Probability"].mean()
    )


    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Employees",
            total_employees
        )

    with col2:

        st.metric(
            "🔴 High Risk",
            high_risk
        )

    with col3:

        st.metric(
            "🟡 Medium Risk",
            medium_risk
        )

    with col4:

        st.metric(
            "🟢 Low Risk",
            low_risk
        )


    st.metric(
        "Average Attrition Probability",
        f"{average_probability * 100:.2f}%"
    )


    # --------------------------------------------------
    # Risk Distribution
    # --------------------------------------------------

    st.subheader("🎯 Employee Risk Distribution")

    risk_distribution = (
        risk_df["Risk_Level"]
        .value_counts()
        .rename_axis("Risk Level")
        .reset_index(name="Employees")
    )

    st.bar_chart(
        risk_distribution.set_index("Risk Level")
    )


    # --------------------------------------------------
    # Actual Attrition
    # --------------------------------------------------

    st.subheader("📈 Actual Attrition Distribution")

    attrition_distribution = (
        processed_df["Attrition"]
        .value_counts()
        .rename_axis("Attrition")
        .reset_index(name="Employees")
    )

    st.bar_chart(
        attrition_distribution.set_index("Attrition")
    )


    # --------------------------------------------------
    # Attrition by Department
    # --------------------------------------------------

    st.subheader("🏢 Attrition by Department")

    department_attrition = pd.crosstab(
        processed_df["Department"],
        processed_df["Attrition"]
    )

    st.bar_chart(
        department_attrition
    )


    # --------------------------------------------------
    # Attrition by Job Role
    # --------------------------------------------------

    st.subheader("💼 Attrition by Job Role")

    role_attrition = pd.crosstab(
        processed_df["JobRole"],
        processed_df["Attrition"]
    )

    st.bar_chart(
        role_attrition
    )


    # --------------------------------------------------
    # Overtime vs Attrition
    # --------------------------------------------------

    st.subheader("⏰ Overtime vs Attrition")

    overtime_attrition = pd.crosstab(
        processed_df["OverTime"],
        processed_df["Attrition"]
    )

    st.bar_chart(
        overtime_attrition
    )


except Exception as e:

    st.error(
        f"Dashboard could not be loaded: {e}"
    )