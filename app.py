# ============================================================
# app.py — CVD RISK PREDICTION API  (UCI Heart Disease Dataset)
# ============================================================
# WHAT CHANGED FROM THE PREVIOUS VERSION:
#   - Dataset: Framingham → UCI Heart Disease (Cleveland)
#   - Features: 14 Framingham fields → 13 UCI fields
#   - Target: 10-year CVD risk → current heart disease detection
#   - No scaling needed (Random Forest, same as before)
#   - Better balanced dataset (54%/46%) → model accuracy: 83.6%
# ============================================================

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the Random Forest model trained on UCI Heart Disease data
model = joblib.load('model/cvd_model_rf.pkl')

print("✅ Random Forest model (UCI Heart Disease) loaded successfully")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    # ── UCI Heart Disease feature order (MUST match training columns) ────
    # ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    #  'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    #
    # cp       → Chest pain type (0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic)
    # trestbps → Resting blood pressure (mmHg)
    # chol     → Serum cholesterol (mg/dL)
    # fbs      → Fasting blood sugar >120 mg/dL (1=yes, 0=no)
    # restecg  → Resting ECG results (0=normal, 1=ST-T abnormality, 2=LV hypertrophy)
    # thalach  → Max heart rate achieved
    # exang    → Exercise-induced angina (1=yes, 0=no)
    # oldpeak  → ST depression induced by exercise
    # slope    → Slope of peak exercise ST segment (0=upsloping, 1=flat, 2=downsloping)
    # ca       → Number of major vessels coloured by fluoroscopy (0–3)
    # thal     → Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)
    try:
        features = [
            float(data['age']),
            float(data['sex']),
            float(data['cp']),
            float(data['trestbps']),
            float(data['chol']),
            float(data['fbs']),
            float(data['restecg']),
            float(data['thalach']),
            float(data['exang']),
            float(data['oldpeak']),
            float(data['slope']),
            float(data['ca']),
            float(data['thal']),
        ]
    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400

    input_array = np.array(features).reshape(1, -1)

    prediction    = model.predict(input_array)[0]
    probability   = model.predict_proba(input_array)[0]
    # In UCI dataset: target=1 means heart disease PRESENT
    heart_disease_prob = round(probability[1] * 100, 1)

    if heart_disease_prob < 30:
        risk_level = "Low"
        message = (
            "Your clinical profile suggests a low likelihood of heart disease. "
            "Keep up regular exercise, a balanced diet, and annual checkups "
            "to maintain your cardiovascular health."
        )
    elif heart_disease_prob < 60:
        risk_level = "Moderate"
        message = (
            "Your profile shows a moderate risk of heart disease. "
            "Consider discussing your chest pain patterns, blood pressure, "
            "and cholesterol levels with your doctor at your next visit."
        )
    else:
        risk_level = "High"
        message = (
            "Your clinical indicators suggest a high likelihood of heart disease. "
            "Please consult a cardiologist as soon as possible. "
            "Early diagnosis and lifestyle changes can make a significant difference."
        )

    return jsonify({
        'prediction': int(prediction),
        'risk_level': risk_level,
        'probability': heart_disease_prob,
        'message': message
    })


if __name__ == '__main__':
    app.run(debug=True)