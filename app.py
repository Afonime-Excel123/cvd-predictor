# ============================================================
# app.py — CVD RISK PREDICTION API
# ============================================================
# This is the Flask backend. Its only jobs are:
#   1. Serve the frontend (HTML page)
#   2. Accept user health data from the form
#   3. Load the saved model and scaler
#   4. Scale the input the same way we did during training
#   5. Run the prediction
#   6. Return the risk result back to the frontend
# ============================================================

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# ── Create the Flask app ────────────────────────────────────
# __name__ tells Flask where to look for templates and static files
app = Flask(__name__)

# ── Load the model and scaler once when the server starts ───
# We load them here (outside any route) so they are loaded
# once and reused for every request. Loading inside the route
# would reload them on every single prediction — very slow.
model  = joblib.load('model/cvd_model.pkl')
scaler = joblib.load('model/scaler.pkl')

print("✅ Model and scaler loaded successfully")


# ── Route 1: Serve the frontend ─────────────────────────────
# When a user visits http://localhost:5000 in their browser,
# Flask returns the index.html file from the templates folder.
@app.route('/')
def home():
    return render_template('index.html')


# ── Route 2: Handle the prediction ──────────────────────────
# This route listens for POST requests at /predict.
# The frontend will send the user's health data here as JSON.
# We process it, run the model, and send back the result.
@app.route('/predict', methods=['POST'])
def predict():

    # Get the JSON data sent from the frontend
    data = request.get_json()

    # ── Extract each field from the incoming data ────────────
    # The order here MUST match the order we trained with:
    # ['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds',
    #  'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol',
    #  'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
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
        # If any field is missing from the request, return an error
        return jsonify({'error': f'Missing field: {str(e)}'}), 400

    # ── Scale the input ──────────────────────────────────────
    # We MUST use the same scaler from training.
    # numpy array with reshape(1, -1) means:
    # "treat this as one row with as many columns as needed"
    input_array = np.array(features).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    # ── Run the prediction ───────────────────────────────────
    # predict() returns 0 or 1
    # predict_proba() returns the probability of each class
    # e.g. [0.72, 0.28] means 72% chance of no CVD, 28% CVD
    prediction    = model.predict(input_scaled)[0]
    probability   = model.predict_proba(input_scaled)[0]
    cvd_risk_prob = round(probability[1] * 100, 1)

    # ── Build the risk level and health message ──────────────
    # Based on probability percentage we assign a risk tier
    # and a personalized message to show the user
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

    # ── Send the result back to the frontend ─────────────────
    # jsonify() converts our Python dictionary to a JSON response
    return jsonify({
        'prediction': int(prediction),
        'risk_level': risk_level,
        'probability': cvd_risk_prob,
        'message': message
    })


# ── Start the server ─────────────────────────────────────────
# debug=True means the server auto-restarts when you save changes
# Only use debug=True in development, never in production
if __name__ == '__main__':
    app.run(debug=True)