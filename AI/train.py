import torch
from pandas import read_csv
from numpy import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pickle import dump

try:
    from AI.model import HealthRiskNet
    from AI.utils import process_habits, boost_habits
except:
    from model import HealthRiskNet
    from utils import process_habits, boost_habits

data = read_csv("AI/data.csv")

data['habits_processed'] = data['habits'].apply(process_habits) 

gender_encoder = LabelEncoder()
cholesterol_encoder = LabelEncoder()
data['gender_encoded'] = gender_encoder.fit_transform(data['gender'])  # Encoded gender
data['cholesterol_encoded'] = cholesterol_encoder.fit_transform(data['cholesterol'])  # Encoded cholesterol


data['health_risk'] = data['health_risk'].str.strip()
health_risk_encoder = LabelEncoder()
data['health_risk_encoded'] = health_risk_encoder.fit_transform(data['health_risk'])  # Encoded the health risk

# Gets all numeric features
numeric_features = data[['age', 'gender_encoded', 'cholesterol_encoded', 'systolic_bp', 'diastolic_bp']].values 
scaler = StandardScaler() 
numeric_features = scaler.fit_transform(numeric_features) # Scales all the numeric values

habits_scaler = StandardScaler()

data['habits_processed'] = habits_scaler.fit_transform(data['habits_processed'].values.reshape(-1, 1))
data['habits_boosted'] = data['habits_processed'].apply(boost_habits)

numeric_features = hstack((numeric_features, data['habits_boosted'].values.reshape(-1, 1)))

x = torch.tensor(numeric_features, dtype=torch.float32) 
y = torch.tensor(data['health_risk_encoded'])

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

classes = health_risk_encoder.classes_

input_dim = x.shape[1]  # Get how many columns
output_dim = len(data['health_risk_encoded'].unique())  # Gets how many classifications

# Saves the cached encoders and scaler
with open('AI/cache/gender_encoder.pkl', 'wb') as f:
    dump(gender_encoder, f) 
with open('AI/cache/cholesterol_encoder.pkl', 'wb') as f:
    dump(cholesterol_encoder, f)
with open('AI/cache/scaler.pkl', 'wb') as f:
    dump(scaler, f)
with open('AI/cache/habits_scaler.pkl', 'wb') as f:
    dump(habits_scaler, f)

model = HealthRiskNet(input_dim=input_dim, output_dim=output_dim)
#model.fit(x_train, y_train, 400)  # Trains the model

with torch.no_grad():
    y_pred = model(x_val)
    for i in range(len(y_pred)):
        predicted = torch.argmax(y_pred[i]).item()
        actual = y_val[i].item()
        if predicted != actual:
            print(f"Data: {x_val[i]} Predicted: {classes[predicted]}, Actual: {classes[actual]}")

