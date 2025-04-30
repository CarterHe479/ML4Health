import torch
import torch.nn as nn
from torchvision import models

class ResNet51(nn.Module):
    def __init__(self):
        super().__init__()
        
        # RGB branch
        self.rgb_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Depth branch
        self.depth_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Modify first conv layer of depth branch to accept 3 channels
        original_conv = self.depth_backbone.conv1
        self.depth_backbone.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        
        # Remove final layers
        self.rgb_backbone.fc = nn.Identity()
        self.depth_backbone.fc = nn.Identity()
        
        # Combined head
        self.regressor = nn.Sequential(
            nn.Linear(4096, 1024),  # 2048*2 (both backbones)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 4)  # mass, fat, carbs, protein
        )
        
    def forward(self, rgb, depth):
        # Process RGB
        rgb_features = self.rgb_backbone(rgb)
        
        # Process depth
        depth_features = self.depth_backbone(depth)
        
        # Combine features
        combined = torch.cat([rgb_features, depth_features], dim=1)
        
        # Regression
        return self.regressor(combined)