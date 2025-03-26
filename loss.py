import torch.nn as nn

class NutritionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, outputs, targets):
        return self.mse_loss(outputs, targets)