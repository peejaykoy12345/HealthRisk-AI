from flask import render_template,url_for
from torch import tensor, float32
from HealthRisk_AI import app
from HealthRisk_AI.forms import InputForm
from AI.predict import predict_risk
from AI.utils import process_habits, boost_habits
from pandas import DataFrame
from numpy import hstack, array
from pickle import load

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = InputForm()
    if form.validate_on_submit():
        with open('AI/cache/gender_encoder.pkl', 'rb') as f:
            gender_encoder = load(f)
        with open('AI/cache/cholesterol_encoder.pkl', 'rb') as f:
            cholesterol_encoder = load(f)
        with open('AI/cache/scaler.pkl', 'rb') as f:
            scaler = load(f)
        with open('AI/cache/habits_scaler.pkl', 'rb') as f:
            habits_scaler = load(f)

        selected_habits = form.habits.data
        habits = ';'.join(selected_habits)
        
        processed_habits = process_habits(habits)

        gender = form.gender.data
        cholesterol = form.cholesterol.data
        gender = gender_encoder.transform([gender])[0]
        cholesterol = cholesterol_encoder.transform([cholesterol])[0]

        age = form.age.data
        systolic_bp = form.systolic_bp.data
        diastolic_bp = form.diastolic_bp.data

        numeric_features = DataFrame([{
            'age': age,
            'gender_encoded': gender,
            'cholesterol_encoded': cholesterol,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
        }]) 
        numeric_features = scaler.transform(numeric_features)

        processed_habits = habits_scaler.transform([[processed_habits]])[0][0]
        boosted_habits = boost_habits(processed_habits)

        numeric_features = hstack((numeric_features, array([[boosted_habits]])))

        data = tensor(numeric_features, dtype=float32)

        prediction, confidence = predict_risk(data)

        recommendations = []

        if age >= 50 and "no exercise" in habits:
            recommendations.append("Being over 50 with no exercise increases health risk. Try adding light daily walks.")

        if "smokes" in habits:
            recommendations.append("Smoking increases risk significantly. Consider quitting or seeking support.")

        if "smokes heavily" in habits:
            recommendations.append("Heavy smoking is linked to high health risk. Seek immediate medical advice.")

        if "no exercise" in habits or "sedentary" in habits:
            recommendations.append("Lack of physical activity increases cardiovascular risk. Start with light movement.")

        if "stressed" in habits:
            recommendations.append("Stress can raise blood pressure and risk. Consider mindfulness, yoga, or counseling.")

        if systolic_bp >= 140 or diastolic_bp >= 90:
            recommendations.append("High blood pressure detected. Consult a doctor for management.")

        if cholesterol == "Very High":
            recommendations.append("Cholesterol levels are dangerously high. A low-fat diet and medication may help.")

        if "drinks heavily" in habits or ("drinks" in habits and age >= 50):
            recommendations.append("Heavy alcohol use at your age can increase heart risk. Reduce consumption.")

        if prediction == "Low":
            recommendations.append("Keep up your healthy lifestyle! Regular activity and no smoking are great.")

        return render_template('result.html', prediction=prediction, confidence=confidence, recommendations=recommendations)

    return render_template('predict.html', form=form)
    