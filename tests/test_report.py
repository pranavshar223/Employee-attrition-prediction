from src.generate_report import generate_employee_report


def test_pdf_report_generation():

    employee_data = {
        "Age": 30,
        "Gender": "Male",
        "Department": "Sales",
        "JobRole": "Sales Executive",
        "BusinessTravel": "Travel_Rarely",
        "OverTime": "Yes"
    }

    probability = 0.63
    risk = "High Risk"

    recommendations = [
        "Review employee workload.",
        "Discuss career progression.",
        "Consider retention actions."
    ]

    pdf = generate_employee_report(
        employee_data,
        probability,
        risk,
        recommendations
    )

    assert pdf is not None

    assert pdf.read(4) == b"%PDF"