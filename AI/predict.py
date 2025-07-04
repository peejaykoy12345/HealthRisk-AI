from torch import load, argmax
from AI.model import HealthRiskNet
from AI.train import input_dim, output_dim

model = HealthRiskNet(input_dim=input_dim, output_dim=output_dim)
model.load_state_dict(load('AI/model.pth')) 
model.eval()

classes = ['Low', 'Medium', 'High']

def predict_risk(x):
    prediction = argmax(model(x))
    return classes[prediction.item()]