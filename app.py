from flask import Flask, render_template, request, send_file
import numpy as np
import joblib
from datetime import datetime
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)
model = joblib.load('model/model.pkl')

last_result = {}

def safe_int(val, default=0):
    return int(val) if val else default

def safe_float(val, default=0.0):
    return float(val) if val else default


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global last_result

    # Basic patient info
    name = request.form.get('name')
    age = safe_int(request.form.get('age'))
    gender = request.form.get('gender')
    bp = safe_int(request.form.get('bp'))
    chol = safe_int(request.form.get('chol'))
    heart_rate = safe_int(request.form.get('heart_rate'))

    exercise = request.form.get('exercise')
    sugar = request.form.get('sugar')

    # Medical features
    cp = safe_int(request.form.get('cp'))

    # DEFAULT VALUES (as you requested)
    restecg = 1
    oldpeak = 1.0
    slope = 1
    ca = 0
    thal = 2

    # Encoding categorical values
    sex = 1 if gender == "Male" else 0
    exang = 1 if exercise == "Yes" else 0
    fbs = 0 if sugar == "Normal" else 1

    # Model input
    data = np.array([[age, sex, cp, bp, chol,
                      fbs, restecg, heart_rate, exang,
                      oldpeak, slope, ca, thal]])

    prob = model.predict_proba(data)[0][1]
    score = round(prob * 100, 2)

    # Risk classification
    if score < 30:
        risk = "Low Risk"
        advice = "Maintain a healthy lifestyle with balanced diet and regular physical activity."
        avoid = "Avoid smoking, junk food, and sedentary habits."
        actions = [
            "Continue regular exercise and stay active.",
            "Keep a balanced diet rich in vegetables and lean protein.",
            "Schedule periodic health screenings."
        ]
        high_risk_actions = []
    elif score < 70:
        risk = "Moderate Risk"
        advice = "Regular health checkups and lifestyle improvements are recommended."
        avoid = "Avoid oily food, stress, and irregular sleep patterns."
        actions = [
            "Adopt a heart-healthy diet low in saturated fat.",
            "Monitor blood pressure and cholesterol frequently.",
            "Increase stress management and sleep quality."
        ]
        high_risk_actions = []
    else:
        risk = "High Risk"
        advice = "Immediate consultation with a cardiologist is strongly recommended."
        avoid = "Avoid high salt, sugar, smoking, and alcohol."
        actions = [
            "See a cardiologist as soon as possible.",
            "Schedule diagnostic tests such as ECG and stress testing.",
            "Reduce sodium intake and stop smoking."
        ]
        high_risk_actions = [
            "Keep emergency contact information available.",
            "Track symptoms daily and note any changes.",
            "Follow prescribed medication and treatment plans strictly."
        ]

    # Store result
    current_date = datetime.now().strftime("%B %d, %Y")
    last_result = {
        "name": name,
        "risk": risk,
        "score": score,
        "advice": advice,
        "avoid": avoid,
        "actions": actions,
        "high_risk_actions": high_risk_actions,
        "current_date": current_date
    }

    return render_template("result.html", **last_result)


@app.route('/download')
def download():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("HEART HEALTH REPORT", styles['Title']))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Patient Name: {last_result.get('name')}", styles['Normal']))
    content.append(Paragraph(f"Risk Level: {last_result.get('risk')}", styles['Normal']))
    content.append(Paragraph(f"Risk Score: {last_result.get('score')}%", styles['Normal']))

    content.append(Spacer(1, 10))
    content.append(Paragraph("Recommendation:", styles['Heading2']))
    content.append(Paragraph(last_result.get('advice'), styles['Normal']))

    content.append(Paragraph("Action Plan:", styles['Heading2']))
    actions_text = '<br/>'.join(['• ' + item for item in last_result.get('actions', [])])
    content.append(Paragraph(actions_text, styles['Normal']))

    content.append(Paragraph("What to Avoid:", styles['Heading2']))
    content.append(Paragraph(last_result.get('avoid'), styles['Normal']))

    if last_result.get('high_risk_actions'):
        content.append(Paragraph("High Risk Intervention Steps:", styles['Heading2']))
        high_risk_text = '<br/>'.join(['• ' + item for item in last_result.get('high_risk_actions', [])])
        content.append(Paragraph(high_risk_text, styles['Normal']))

    content.append(Spacer(1, 10))
    content.append(Paragraph("Note: AI-based assessment. Not a medical diagnosis.", styles['Italic']))

    doc.build(content)
    buffer.seek(0)

    return send_file(buffer,
                     as_attachment=True,
                     download_name="Heart_Report.pdf",
                     mimetype='application/pdf')


if __name__ == "__main__":
    app.run(debug=True)