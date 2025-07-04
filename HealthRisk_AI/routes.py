from flask import render_template,url_for
from torch import tensor, float32
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer, LabelEncoder, StandardScaler
from HealthRisk_AI import app
from HealthRisk_AI.forms import InputForm
from AI.predict import predict_risk
from pandas import DataFrame
from numpy import hstack

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = InputForm()
    if form.validate_on_submit():
        selected_habits = form.habits.data
        habits = ';'.join(selected_habits)
        vectorizer = TfidfVectorizer(max_features=1000)
        tfid_features = vectorizer.fit_transform([habits])  
        normalizer = Normalizer()
        tfid_features = normalizer.fit_transform(tfid_features)

        gender = form.gender.data
        cholesterol = form.cholesterol.data
        gender_encoder = LabelEncoder()
        cholesterol_encoder = LabelEncoder()
        gender = gender_encoder.fit_transform([gender])[0]
        cholesterol = cholesterol_encoder.fit_transform([cholesterol])[0]

        age = form.age.data
        systolic_bp = form.systolic_bp.data
        diastolic_bp = form.diastolic_bp.data

        numeric_features = DataFrame([{
            'age': age,
            'gender_encoded': gender,
            'cholesterol_encoded': cholesterol,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp
        }]) 
        scaler = StandardScaler() 
        numeric_features = scaler.fit_transform(numeric_features)

        data = hstack([numeric_features, tfid_features.toarray()]) 
        data = tensor(data, dtype=float32)

        prediction = predict_risk(data)

        return render_template('result.html', prediction=prediction)

    return render_template('predict.html', form=form)
    