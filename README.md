# Healthcare Disease Risk Prediction App

## Overview

This project is a **machine learning–powered healthcare risk prediction web application** built with  Flask . The system predicts whether a patient has a high likelihood of disease risk based on medical and lifestyle features.

The application also integrates **Google Gemini AI** to generate additional insights and explanations for predictions.

The goal of this project is to provide a simple interface where users can input health metrics and receive:

* Disease risk prediction
* AI‑generated medical insights



# Features

* Machine Learning model for disease risk prediction
* Flask web application
* Gemini AI integration for health insights
* User-friendly web interface
* Preprocessing pipeline saved with the trained model



# Project Structure


app/
│
├── app.py                  # Flask application
├── Training model.py       # Model training script
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys)
│
├── models/
│   └── model_pipeline.joblib
│
├── templates/              # HTML templates
├── static/                 # CSS / JS / assets
├── data/                   # Dataset used for training
│
└── README.md


# Tech Stack

## Backend

* Python
* Flask

## Machine Learning

* Scikit‑learn
* Pandas
* Joblib

## AI Integration

* Google Gemini API

## Frontend 

* HTML
* CSS



# Installation

## 1. Clone the repository

## 2. Create a virtual environment

python -m venv venv

## Activate environment

### Windows
venv\Scripts\activate
### Mac/Linux
source venv/bin/activate


## 3. Install dependencies


pip install -r requirements.txt

# Environment Variables

Create a `.env` file in the root directory.


GEMINI_API_KEY=your_google_gemini_api_key


You can obtain the API key from:

[https://ai.google.dev](https://ai.google.dev)

# Running the Application


python app.py

The application will start at:

http://127.0.0.1:5000

# Model Training

To retrain the model:
python "Training model.py"

## This script will:

1. Load the dataset
2. Preprocess features
3. Train the machine learning model
4. Save the pipeline as


models/model_pipeline.joblib




# Input Features

## Numerical Features

* age
* bmi
* systolic_bp
* diastolic_bp
* cholesterol
* blood_glucose
* heart_rate
* weight_kg
* height_cm
* triglycerides
* creatinine
* hemoglobin
* body_temp_c
* wbc_count
* platelet_count
* sleep_hours

## Categorical Features

* gender
* smoking_status
* region
* insurance_type
* family_history



# Prediction Output

The model returns:

| Prediction | Meaning                     |
| ---------- | --------------------------- |
| 0          | No significant disease risk |
| 1          | High likelihood of disease  |

Additionally, **Gemini AI generates personalized insights** based on the prediction.

# Future Improvements

* Add authentication system
* Deploy using Docker
* Add REST API endpoints
* Improve model performance
* Add visualization dashboard


# License

This project is licensed under the **MIT License**.



 ## Happy Coding

For questions or collaboration, feel free to reach out.
