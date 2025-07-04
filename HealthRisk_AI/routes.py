from flask import render_template,url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer, LabelEncoder, StandardScaler
from HealthRisk_AI import app
from HealthRisk_AI.forms import InputForm
from AI.predict import predict
from pandas import DataFrame

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    form = InputForm()
    if form.validate_on_submit():
        selected_habits = form.habits.data
        habits = ';'.join(selected_habits)
        vectorizer = TfidfVectorizer(max_features=1000)
        tfid_features = vectorizer.fit_transform(habits.astype(str))  
        normalizer = Normalizer()
        tfid_features = normalizer.fit_transform(tfid_features)

        gender = form.gender.data
        cholesterol = form.cholesterol.data
        gender_encoder = LabelEncoder()
        cholesterol_encoder = LabelEncoder()
        gender = gender_encoder.fit_transform(gender) 
        cholesterol = cholesterol_encoder.fit_transform(cholesterol)

        age = form.age.data
        systolic_bp = form.systolic_bp.data
        diastolic_bp = form.diastolic_bp.data

        numeric_features = DataFrame([{
            'age': age,
            'gender_encoded': gender,
            'cholesterol_encoded': cholesterol,
            'systolic_bp': systolic_bp,
            'diaslostic_bp': diastolic_bp
        }]) 
        scaler = StandardScaler() 
        numeric_features = scaler.fit_transform(numeric_features)

    return render_template('predict.html', form=form)