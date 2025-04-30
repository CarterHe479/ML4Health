from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import Nutrition5KDataset
from train import Trainer
import logging
from utils import setup_logging, get_device
from loss import NutritionLoss, MSELoss, MSEMultiTaskLoss
from multi_task_model import MultiTaskModel, get_base_model
from rgb_depth_model import ResNet51

def get_transforms():
    # RGB normalization (ImageNet stats)
    rgb_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Depth normalization (adjust based on your depth data)
    depth_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                           std=[0.5, 0.5, 0.5])
    ])
    
    return rgb_transform, depth_transform

def main():
    setup_logging()
    device = get_device()
    # Data transforms
    rgb_transform, depth_transform = get_transforms()
    
    # Load datasets
    logging.info("Loading datasets...")
    root_dir = '../nutrition5k_dataset'
    
    train_dataset = Nutrition5KDataset(
        root_dir=root_dir,
        split='train',
        rgb_transform=rgb_transform,
        depth_transform=depth_transform
    )
    
    val_dataset = Nutrition5KDataset(
        root_dir=root_dir,
        split='val',
        rgb_transform=rgb_transform,
        depth_transform=depth_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Initialize model and trainer
    model = ResNet51().to(device)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        loss = MSELoss(),
        learning_rate=1e-4,
        weight_decay=1e-5,
        use_depth=True,
        work_dir = 'rgb_depth_model'
    )
    
    # Train the model
    logging.info("Starting training...")
    trainer.train(num_epochs=500)
    logging.info("Training completed")

if __name__ == '__main__':
    main()