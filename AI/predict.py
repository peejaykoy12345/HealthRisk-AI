from torch import load, argmax, no_grad
from torch.nn.functional import softmax
from AI.model import HealthRiskNet
from AI.train import input_dim, output_dim, classes

model = HealthRiskNet(input_dim=input_dim, output_dim=output_dim)
model.load_state_dict(load('AI/model.pth')) 
model.eval()

def predict_risk(x):
    with no_grad():
        print(x)
        tensor_prediction = model(x)
        probabilities = softmax(tensor_prediction, dim=1)
        prediction_index = (argmax(probabilities, dim=1)).item()
        label = classes[prediction_index]
        confidence = probabilities[0][prediction_index].item()

    return label, round(confidence * 100, 1)