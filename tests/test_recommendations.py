from src.recommendations import generate_recommendations


def test_high_risk_employee_gets_recommendation():

    employee_data = {
        "OverTime": "Yes",
        "JobSatisfaction": 1,
        "EnvironmentSatisfaction": 1,
        "RelationshipSatisfaction": 1,
        "WorkLifeBalance": 1,
        "YearsSinceLastPromotion": 6,
        "YearsInCurrentRole": 8,
        "PercentSalaryHike": 10,
        "TrainingTimesLastYear": 1,
        "StockOptionLevel": 0
    }

    recommendations = generate_recommendations(
        employee_data,
        "High Risk"
    )

    assert isinstance(recommendations, list)

    assert len(recommendations) > 0