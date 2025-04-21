import torch.nn as nn
import torch
from utils import get_device

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.mse_loss = nn.MSELoss()
    
    def forward(self, outputs, targets):
        diff = outputs - targets
        diff[:, 0] = diff[:, 0] / 10
        # Return mean of squared differences
        return (diff ** 2).mean()

class MSEMultiTaskLoss(nn.Module):
    def __init__(self, portion_independent=False):
        super().__init__()
        # self.mse_loss = nn.MSELoss()
        self.portion_independent = portion_independent
    
    def forward(self, outputs, targets):
        if self.portion_independent:
            outputs = torch.cat([outputs['mass'], outputs['fat'], outputs['carbs'], outputs['protein']], dim=1)
            diff = outputs - targets
        else:
            outputs = torch.cat([outputs['fat'], outputs['carbs'], outputs['protein']], dim=1)
            diff = outputs - targets[:, 1:]
        diff[:, 0] = diff[:, 0] / 10
        # Return mean of squared differences
        return (diff ** 2).mean()
    
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