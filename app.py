from flask import Flask, render_template, request
import joblib
import pandas as pd
import hashlib
import os
from dotenv import load_dotenv
import google.generativeai as genai
from google import genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

gemini_cache = {}

try:
    model = joblib.load("models/model_pipeline.joblib")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

app = Flask(__name__)

prediction_labels = {
    0: "No significant disease risk detected. Keep maintaining a healthy lifestyle!",
    1: "High likelihood of disease detected. Please consult a healthcare professional promptly."
}

numerical_features = [
    'age', 'bmi', 'systolic_bp', 'diastolic_bp', 'cholesterol',
    'blood_glucose', 'heart_rate', 'weight_kg', 'height_cm',
    'triglycerides', 'creatinine', 'hemoglobin', 'body_temp_c',
    'wbc_count', 'platelet_count', 'sleep_hours'
]

categorical_features = [
    'gender', 'smoking_status', 'region', 'insurance_type', 'family_history'
]

def generate_insights_with_gemini(prediction, probabilities, patient_data):

    try:
        cache_input = f"{prediction}-{probabilities}-{patient_data}"
        cache_key = hashlib.md5(cache_input.encode()).hexdigest()
        if cache_key in gemini_cache:
            return gemini_cache[cache_key]

        prompt = f"""
Predicted condition: {prediction}
Probabilities: {probabilities}
Patient data: {patient_data}

Provide simple, empathetic health insights related to the input medical values.
Also include lifestyle recommendations. Do NOT provide medical diagnosis.
Use bullet points, be supportive and clear.
"""
        model_gemini = genai.GenerativeModel("gemini-2.0-flash")
        response = model_gemini.generate_content(prompt)
        insights = response.text.strip() if response and response.text else \
                   "Unable to generate insights at this time. Please consult a healthcare professional."
        gemini_cache[cache_key] = insights
        return insights
    except Exception as e:
        print("Gemini API error:", e)
        return "Unable to generate insights at this time. Please consult a healthcare professional."


@app.route("/")
def index():
    return render_template("index.html", errors={}, input_data={})


@app.route("/predict", methods=["POST"])
def predict():
    errors = {}
    patient_data = {}

    for field in numerical_features:
        value = request.form.get(field, '')
        try:
            val = float(value)
            if val <= 0:
                errors[field] = f"{field.replace('_',' ').title()} must be positive."
            patient_data[field] = val
        except ValueError:
            errors[field] = f"Invalid value for {field.replace('_',' ').title()}."

    for field in categorical_features:
        value = request.form.get(field, '')
        if not value:
            errors[field] = f"Invalid value for {field.replace('_',' ').title()}."
        patient_data[field] = value

    if errors:
        return render_template("index.html", errors=errors, input_data=patient_data)

    try:
        X = pd.DataFrame([patient_data])

        probabilities = model.predict_proba(X)[0]
        raw_prediction = int(model.predict(X)[0])
        prediction_text = prediction_labels.get(raw_prediction, str(raw_prediction))

        insights = generate_insights_with_gemini(prediction_text, probabilities.tolist(), patient_data)

        return render_template("result.html",
                               prediction=prediction_text,
                               probabilities=probabilities.tolist(),
                               classes=model.classes_.tolist(),
                               insights=insights,
                               patient_data=patient_data,
                               errors={})
    except Exception as e:
        return render_template("index.html",
                               errors={"general": f"Error during prediction: {e}"},
                               input_data=patient_data)


if __name__ == "__main__":
    app.run(debug=True)
