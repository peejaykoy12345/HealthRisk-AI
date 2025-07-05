import torch.nn as nn
import torch.optim as optim
from torch import save

class HealthRiskNet(nn.Module):
    def __init__(self, input_dim=6, output_dim=3):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)
    
    def fit(self, x, y, epochs):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            predictions = self.model(x) 
            loss = criterion(predictions, y)

            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 

            if (epoch + 1) % 50 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
                save(self.state_dict(), 'AI/model.pth')
    