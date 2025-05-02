import torch
import torch.nn as nn
import torchvision.models as models

class CrossAttentionNutritionModel(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, num_heads=4, mlp_hidden=256):
        super(CrossAttentionNutritionModel, self).__init__()

        # RGB+Side branch: ResNet with 6 input channels
        if backbone == 'resnet18':
            self.rgb_side_backbone = models.resnet18(pretrained=pretrained)
            self.depth_backbone = models.resnet18(pretrained=pretrained)
            backbone_out_dim = 512
        elif backbone == 'resnet34':
            self.rgb_side_backbone = models.resnet34(pretrained=pretrained)
            self.depth_backbone = models.resnet34(pretrained=pretrained)
            backbone_out_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Modify conv1 to accept 6-channel RGB+side input
        self.rgb_side_backbone.conv1 = nn.Conv2d(
            6, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Modify conv1 to accept 1-channel Depth input
        self.depth_backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove classification heads, keep feature extractor
        self.rgb_side_backbone_fc = nn.Identity()
        self.depth_backbone_fc = nn.Identity()

        # Cross-Attention: embed_dim = backbone output dim
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=backbone_out_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # MLP: from fused feature to 4 regression outputs
        self.mlp = nn.Sequential(
            nn.Linear(backbone_out_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 4)  # [fat_g, carb_g, protein_g, kcal]
        )

    def forward(self, rgb_side, depth):
        """
        Args:
            rgb_side: Tensor [B, 6, H, W]
            depth: Tensor [B, 1, H, W]
        Returns:
            out: Tensor [B, 4]
        """

        # Extract features
        rgb_feat = self._extract_feat(self.rgb_side_backbone, rgb_side)   # [B, 512, H/32, W/32]
        depth_feat = self._extract_feat(self.depth_backbone, depth)       # [B, 512, H/32, W/32]

        # Flatten spatial dims: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = rgb_feat.shape
        rgb_seq = rgb_feat.flatten(2).permute(0, 2, 1)    # [B, H*W, C]
        depth_seq = depth_feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        # Cross Attention: query=rgb, key=value=depth
        attn_out, _ = self.cross_attention(
            query=rgb_seq,
            key=depth_seq,
            value=depth_seq
        )  # [B, H*W, C]

        # Global average pooling across sequence
        fused_feat = attn_out.mean(dim=1)  # [B, C]

        # Predict nutrition + kcal
        out = self.mlp(fused_feat)  # [B, 4]
        return out

    @staticmethod
    def _extract_feat(backbone, x):
        """
        Extract features from ResNet backbone up to last conv layer.
        """
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)

        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        x = backbone.layer4(x)  # [B, 512, H/32, W/32]
        return x
