import torch.nn as nn
import torch
from utils import get_device

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, outputs, targets):
        return self.mse_loss(outputs, targets)

class NutritionLoss(nn.Module):
    def __init__(self, initial_scales=None):
        super().__init__()
        if initial_scales is None:
            initial_scales = torch.tensor([1/500.0, 1/50.0, 1/100.0, 1/50.0])
        self.scales = nn.Parameter(initial_scales.to(get_device()))
        
    def forward(self, predictions, targets):
        # Scale both predictions and targets
        scaled_pred = predictions * self.scales
        scaled_target = targets * self.scales
        
        # Then compute loss (MSE or your preferred loss)
        return nn.functional.mse_loss(scaled_pred, scaled_target)