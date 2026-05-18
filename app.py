# ============================================================
# app.py — CVD RISK PREDICTION API
# ============================================================
# This is the Flask backend. Its only jobs are:
#   1. Serve the frontend (HTML page)
#   2. Accept user health data from the form
#   3. Load the saved Random Forest model
#   4. Run the prediction (no scaling needed for RF)
#   5. Return the risk result back to the frontend
#
# WHAT CHANGED FROM THE ORIGINAL:
#   - model file is now cvd_model_rf.pkl (Random Forest)
#   - We no longer scale the input before predicting.
#     Random Forest makes decisions using split thresholds
#     ("is age > 55?") — the actual magnitude of numbers
#     doesn't matter, so scaling has zero effect on it.
#     Removing it keeps the pipeline cleaner and correct.
# ============================================================

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# ── Create the Flask app ────────────────────────────────────
app = Flask(__name__)

# ── Load the Random Forest model once at startup ────────────
# We no longer need to load the scaler — RF doesn't use it.
# Loading outside the route means it happens once, not on
# every single request (which would be very slow).
model = joblib.load('model/cvd_model_rf.pkl')

print("✅ Random Forest model loaded successfully")


# ── Route 1: Serve the frontend ─────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


# ── Route 2: Handle the prediction ──────────────────────────
@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    # ── Extract each field from the incoming data ────────────
    # The order here MUST match the column order we trained with:
    # ['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds',
    #  'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol',
    #  'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    #
    # Note: 'education' was dropped during training, so we
    # do NOT include it here either.
    try:
        features = [
            float(data['male']),
            float(data['age']),
            float(data['currentSmoker']),
            float(data['cigsPerDay']),
            float(data['BPMeds']),
            float(data['prevalentStroke']),
            float(data['prevalentHyp']),
            float(data['diabetes']),
            float(data['totChol']),
            float(data['sysBP']),
            float(data['diaBP']),
            float(data['BMI']),
            float(data['heartRate']),
            float(data['glucose'])
        ]
    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400

    # ── Run the prediction ───────────────────────────────────
    # reshape(1, -1) → treat this as one row with N columns.
    # We pass it directly to the model — no scaling needed.
    #
    # predict()       → returns 0 or 1 (the final decision)
    # predict_proba() → returns probability for each class
    #   e.g. [0.68, 0.32] means 68% no CVD, 32% CVD risk
    input_array = np.array(features).reshape(1, -1)

    prediction    = model.predict(input_array)[0]
    probability   = model.predict_proba(input_array)[0]
    cvd_risk_prob = round(probability[1] * 100, 1)

    # ── Build the risk level and health message ──────────────
    if cvd_risk_prob < 20:
        risk_level = "Low"
        message = (
            "Your cardiovascular risk appears low. "
            "Keep maintaining a healthy lifestyle — "
            "regular exercise, balanced diet, and routine checkups."
        )
    elif cvd_risk_prob < 50:
        risk_level = "Moderate"
        message = (
            "You have a moderate cardiovascular risk. "
            "Consider reducing salt intake, exercising regularly, "
            "and scheduling a checkup with your doctor."
        )
    else:
        risk_level = "High"
        message = (
            "Your cardiovascular risk is high. "
            "Please consult a doctor as soon as possible. "
            "Lifestyle changes and medical guidance are strongly advised."
        )

    return jsonify({
        'prediction': int(prediction),
        'risk_level': risk_level,
        'probability': cvd_risk_prob,
        'message': message
    })


# ── Start the server ─────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)