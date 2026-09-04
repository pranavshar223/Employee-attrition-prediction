from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER


def generate_employee_report(
    employee_data,
    probability,
    risk,
    recommendations
):
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    heading_style = styles["Heading2"]
    normal_style = styles["BodyText"]

    story = []

    # Title
    story.append(
        Paragraph(
            "Employee Attrition Risk Report",
            title_style
        )
    )

    story.append(Spacer(1, 20))

    # Employee Profile
    story.append(
        Paragraph(
            "Employee Profile",
            heading_style
        )
    )

    profile_data = [
        ["Attribute", "Value"],
        ["Age", str(employee_data.get("Age", "N/A"))],
        ["Gender", str(employee_data.get("Gender", "N/A"))],
        ["Department", str(employee_data.get("Department", "N/A"))],
        ["Job Role", str(employee_data.get("JobRole", "N/A"))],
        ["Business Travel", str(employee_data.get("BusinessTravel", "N/A"))],
        ["OverTime", str(employee_data.get("OverTime", "N/A"))],
    ]

    profile_table = Table(
        profile_data,
        colWidths=[180, 300]
    )

    profile_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 8),
        ])
    )

    story.append(profile_table)
    story.append(Spacer(1, 20))

    # Prediction
    story.append(
        Paragraph(
            "Prediction",
            heading_style
        )
    )

    prediction_data = [
        ["Metric", "Result"],
        [
            "Attrition Probability",
            f"{probability * 100:.2f}%"
        ],
        ["Risk Level", risk],
    ]

    prediction_table = Table(
        prediction_data,
        colWidths=[180, 300]
    )

    prediction_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 8),
        ])
    )

    story.append(prediction_table)
    story.append(Spacer(1, 20))

    # Risk Alert
    story.append(
        Paragraph(
            "Risk Assessment",
            heading_style
        )
    )

    if risk == "High Risk":
        alert = (
            "HIGH ATTRITION RISK: Immediate HR attention "
            "and retention discussion are recommended."
        )
    elif risk == "Medium Risk":
        alert = (
            "MEDIUM ATTRITION RISK: Follow-up with the "
            "employee is recommended."
        )
    else:
        alert = (
            "LOW ATTRITION RISK: Continue normal employee "
            "engagement and monitoring."
        )

    story.append(
        Paragraph(alert, normal_style)
    )

    story.append(Spacer(1, 20))

    # HR Recommendations
    story.append(
        Paragraph(
            "HR Recommendations",
            heading_style
        )
    )

    for recommendation in recommendations:
        story.append(
            Paragraph(
                f"• {recommendation}",
                normal_style
            )
        )
        story.append(Spacer(1, 6))

    document.build(story)

    buffer.seek(0)

    return buffer