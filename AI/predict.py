from torch import load, argmax
from model import HealthRiskNet

model = HealthRiskNet()
model.load_state_dict(load('AI/model.pth')) 
model.eval()

classes = ['Low', 'Medium', 'High']

def predict_risk(x):
    prediction = argmax(model(x))
    return classes[prediction.item()]