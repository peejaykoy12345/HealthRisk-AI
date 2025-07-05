from torch import load, argmax
from AI.model import HealthRiskNet
from AI.train import input_dim, output_dim, classes

model = HealthRiskNet(input_dim=input_dim, output_dim=output_dim)
model.load_state_dict(load('AI/model.pth')) 
model.eval()

def predict_risk(x):
    print(x)
    tensor_prediction = model(x)
    prediction = argmax(tensor_prediction)
    label = classes[prediction.item()]
    print(tensor_prediction, " | ", prediction, " | ", label)

    return label