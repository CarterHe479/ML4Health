import torch.nn as nn
import torchvision.models as models

# For the base model, we use a pretrained ResNet18 and remove its final layer.
def get_base_model():
    base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    modules = list(base.children())[:-1]  # remove the final fc layer
    base_model = nn.Sequential(*modules)
    # Freeze parameters
    for param in base_model.parameters():
        param.requires_grad = False
    return base_model

# add branch for different categories of nutrition
class Branch(nn.Module):
    def __init__(self, in_features):
        super(Branch, self).__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

# Multi-task model that attaches branches for each output.
class MultiTaskModel(nn.Module):
    def __init__(self, base_model, in_features, portion_independent=False):
        super(MultiTaskModel, self).__init__()
        self.base_model = base_model
        self.portion_independent = portion_independent
        self.protein_branch = Branch(in_features)
        self.fat_branch = Branch(in_features)
        self.carbs_branch = Branch(in_features)
        if not portion_independent:
            self.mass_branch = Branch(in_features)

    def forward(self, x):
        features = self.base_model(x)  # e.g., shape (batch, 512, 1, 1)
        features = features.view(features.size(0), -1)  # flatten to (batch, 512)
        protein = self.protein_branch(features)
        fat = self.fat_branch(features)
        carbs = self.carbs_branch(features)
        outputs = {'protein': protein, 'fat': fat, 'carbs': carbs}
        if not self.portion_independent:
            mass = self.mass_branch(features)
            outputs['mass'] = mass
        return outputs
    

# example of use of the models
# base_model = get_base_model()
# in_features = 512  # for resnet18
# direct_regression = MultiTaskModel(base_model, in_features, portion_independent=False).to(device)
# portion_independent_model = MultiTaskModel(base_model, in_features, portion_independent=True).to(device)

# sample input
# dataloader of dataset (food RGB images as input and nutrition amount as output)