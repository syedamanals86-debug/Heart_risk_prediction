# Heart Risk Prediction App

A Flask web application that predicts heart disease risk using a trained machine learning model. Users can enter health and lifestyle details, get a risk prediction, view personalized recommendations, and download a professional PDF report.

## Key Features

- Heart disease risk classification using a `RandomForestClassifier`
- Responsive Flask UI with patient input form
- Detailed result page with risk category, score, and actionable guidance
- Downloadable PDF summary report
- Recommendations for maintaining heart health and next steps for high-risk users

## Project Structure

- `app.py` — Flask application and prediction routes
- `train.py` — training script for the model
- `data/heart.csv` — dataset used for training
- `model/model.pkl` — saved Random Forest model
- `templates/` — HTML templates for the input and result pages
- `static/style.css` — shared application styling

## Requirements

The app uses the following Python packages:

- `flask`
- `pandas`
- `scikit-learn`
- `joblib`
- `reportlab`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run the App

From the project root:

```bash
python app.py
```

Then open the local URL shown in the terminal (usually `http://127.0.0.1:5000/`).

## How to Use

1. Open the app in your browser.
2. Enter the patient details and health metrics.
3. Submit the form to view the risk prediction.
4. Download the PDF report if needed.

## Model Details

- The production model is a `RandomForestClassifier` trained in `train.py`.
- The model is saved to `model/model.pkl` and then loaded by `app.py`.
- A logistic regression model is included as a training baseline, but the app uses the Random Forest model for inference.

## Notes

- The app is for informational purposes only and should not replace professional medical advice.
- For high-risk results, review the provided recommendations and consult a doctor.
