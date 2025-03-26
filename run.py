import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import Nutrition5KDataset
from cnn_model import NutritionCNN
from train import Trainer
import logging

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    
    device = torch.accelerator.current_accelerator()
    logging.info(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    logging.info("Loading datasets...")
    root_dir = '/Users/liwuchen/Documents/nutrition5k_dataset'
    
    train_dataset = Nutrition5KDataset(
        root_dir=root_dir,
        split='train',
        transform=transform
    )
    
    val_dataset = Nutrition5KDataset(
        root_dir=root_dir,
        split='val',
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model and trainer
    model = NutritionCNN()
    trainer = Trainer(model, train_loader, val_loader, device=device)
    
    # Train the model
    logging.info("Starting training...")
    trainer.train(num_epochs=20)
    logging.info("Training completed")

if __name__ == '__main__':
    main()