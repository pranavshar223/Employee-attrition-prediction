from src.predict import predict_employee


def test_prediction_returns_valid_result():

    employee_data = {
        "Age": 30,
        "Gender": "Male",
        "MaritalStatus": "Single",
        "Department": "Sales",
        "JobRole": "Sales Executive",
        "BusinessTravel": "Travel_Rarely",
        "Education": 3,
        "EducationField": "Marketing",
        "DailyRate": 800,
        "HourlyRate": 60,
        "MonthlyIncome": 5000,
        "EnvironmentSatisfaction": 3,
        "JobSatisfaction": 3,
        "RelationshipSatisfaction": 3,
        "JobLevel": 2,
        "JobInvolvement": 3,
        "TotalWorkingYears": 5,
        "YearsAtCompany": 3,
        "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 2,
        "NumCompaniesWorked": 2,
        "TrainingTimesLastYear": 2,
        "DistanceFromHome": 5,
        "PercentSalaryHike": 15,
        "StockOptionLevel": 1,
        "PerformanceRating": 3,
        "WorkLifeBalance": 3,
        "OverTime": "No",

        # Engineered features
        "DailyHours": 8,
        "MonthlyWorkingHours": 176,
        "Salary_per_hour": 5000 / 176
    }

    probability, risk = predict_employee(employee_data)

    assert 0 <= probability <= 1

    assert risk in [
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]