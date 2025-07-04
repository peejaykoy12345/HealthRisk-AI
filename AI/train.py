import torch
from AI.model import HealthRiskNet
from pandas import read_csv
from numpy import hstack, array
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer, LabelEncoder, StandardScaler

data = read_csv("AI/data.csv")

vectorizer = TfidfVectorizer(max_features=1000)
tfid_features = vectorizer.fit_transform(data["habits"].astype(str))  # Vectorize
normalizer = Normalizer()
tfid_features = normalizer.fit_transform(tfid_features)  # Normalize the tfid features

gender_encoder = LabelEncoder()
cholesterol_encoder = LabelEncoder()
data['gender_encoded'] = gender_encoder.fit_transform(data['gender'])  # Encoded gender
data['cholesterol_encoded'] = cholesterol_encoder.fit_transform(data['cholesterol'])  # Encoded cholesterol

health_risk_encoder = LabelEncoder()
data['health_risk_encoded'] = health_risk_encoder.fit_transform(data['health_risk'])  # Encoded the health risk

# Gets all numeric features
numeric_features = data[['age', 'gender_encoded', 'cholesterol_encoded', 'systolic_bp', 'diastolic_bp']].values 
scaler = StandardScaler() 
numeric_features = scaler.fit_transform(numeric_features)  # Scales all the numeric values

x = hstack([numeric_features, tfid_features.toarray()])  # Combined them to one hstack
x = torch.tensor(x, dtype=torch.float32)  # Converted to tensor
y = torch.tensor(data['health_risk_encoded'])

input_dim = x.shape[1]  # Get how many columns
output_dim = len(data['health_risk_encoded'].unique())  # Gets how many classifications

import pickle

with open('AI/cache/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('AI/cache/normalizer.pkl', 'wb') as f:
    pickle.dump(normalizer, f)
with open('AI/cache/gender_encoder.pkl', 'wb') as f:
    pickle.dump(gender_encoder, f)
with open('AI/cache/cholesterol_encoder.pkl', 'wb') as f:
    pickle.dump(cholesterol_encoder, f)
with open('AI/cache/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

model = HealthRiskNet(input_dim=input_dim, output_dim=output_dim)
#model.load_state_dict(torch.load('model.pth'))  # Loads the trained model
#model.eval()  # Puts model into predicting mode
#model.fit(x, y, 300)  # Trains the model

test_row = { 
    'age': 45,
    'gender': 'M',
    'cholesterol': 'Normal',
    'systolic_bp': 130,
    'diastolic_bp': 85,
    'habits': "gym member"
}

test_gender = gender_encoder.transform([test_row['gender']])[0]
test_chol = cholesterol_encoder.transform([test_row['cholesterol']])[0]

test_numeric = array([[test_row['age'], test_gender, test_chol, test_row['systolic_bp'], test_row['diastolic_bp']]])
test_numeric = scaler.transform(test_numeric)

test_tfidf = vectorizer.transform([test_row['habits']])
test_tfidf = normalizer.transform(test_tfidf)

test_features = hstack([test_numeric, test_tfidf.toarray()])
test_tensor = torch.tensor(test_features, dtype=torch.float32)

classes = ['Low', 'Medium', 'High']
prediction = torch.argmax(model(test_tensor))  # Predicts the outcome and gets the index using torch.argmax
print(classes[prediction.item()])
