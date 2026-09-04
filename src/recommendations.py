def generate_recommendations(employee_data, risk):
    """
    Generate HR recommendations based on employee attributes
    and predicted attrition risk.
    """

    recommendations = []

    # --------------------------------------------------
    # High / Medium Risk
    # --------------------------------------------------

    if risk == "High Risk":
        recommendations.append(
            "Schedule a retention discussion with the employee."
        )

    elif risk == "Medium Risk":
        recommendations.append(
            "Consider a follow-up discussion to understand "
            "the employee's concerns."
        )

    # --------------------------------------------------
    # Overtime
    # --------------------------------------------------

    if employee_data["OverTime"] == "Yes":
        recommendations.append(
            "Review the employee's overtime workload and "
            "consider improving workload distribution."
        )

    # --------------------------------------------------
    # Job Satisfaction
    # --------------------------------------------------

    if employee_data["JobSatisfaction"] <= 2:
        recommendations.append(
            "Discuss factors affecting job satisfaction "
            "and identify possible improvements."
        )

    # --------------------------------------------------
    # Environment Satisfaction
    # --------------------------------------------------

    if employee_data["EnvironmentSatisfaction"] <= 2:
        recommendations.append(
            "Review the employee's work environment and "
            "identify potential workplace concerns."
        )

    # --------------------------------------------------
    # Relationship Satisfaction
    # --------------------------------------------------

    if employee_data["RelationshipSatisfaction"] <= 2:
        recommendations.append(
            "Assess workplace relationships and consider "
            "manager or team support."
        )

    # --------------------------------------------------
    # Work-Life Balance
    # --------------------------------------------------

    if employee_data["WorkLifeBalance"] <= 2:
        recommendations.append(
            "Review work-life balance and consider "
            "flexible working or workload adjustments."
        )

    # --------------------------------------------------
    # Career Growth
    # --------------------------------------------------

    if employee_data["YearsSinceLastPromotion"] >= 5:
        recommendations.append(
            "Discuss career progression and potential "
            "promotion opportunities."
        )

    # --------------------------------------------------
    # Current Role
    # --------------------------------------------------

    if employee_data["YearsInCurrentRole"] >= 7:
        recommendations.append(
            "Discuss opportunities for role growth, "
            "new responsibilities, or internal mobility."
        )

    # --------------------------------------------------
    # Salary
    # --------------------------------------------------

    if employee_data["PercentSalaryHike"] <= 12:
        recommendations.append(
            "Review compensation growth and salary progression."
        )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    if employee_data["TrainingTimesLastYear"] <= 1:
        recommendations.append(
            "Consider providing additional training and "
            "professional development opportunities."
        )

    # --------------------------------------------------
    # Stock Options
    # --------------------------------------------------

    if employee_data["StockOptionLevel"] == 0:
        recommendations.append(
            "Consider reviewing available employee benefits "
            "and retention incentives."
        )

    # --------------------------------------------------
    # Default Recommendation
    # --------------------------------------------------

    if not recommendations:
        recommendations.append(
            "Continue regular employee engagement and "
            "performance monitoring."
        )

    return recommendations