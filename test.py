import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import Nutrition5KDataset
from model import ResNet51
from loss import NutritionLoss, MSELoss
import logging
import numpy as np
from utils import setup_logging, get_device

def main():
    setup_logging()
    device = get_device()
    
    # Load model
    model = ResNet51().to(device)
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logging.info(f"Loaded model from epoch {checkpoint['epoch']} with val loss {checkpoint['val_loss']:.4f}")
    
    # Get transforms (same as training)
    rgb_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    depth_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load test data
    test_dataset = Nutrition5KDataset(
        root_dir='/Users/liwuchen/Documents/nutrition5k_dataset',
        split='test',
        rgb_transform=rgb_transform,
        depth_transform=depth_transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Test the model
    # criterion = NutritionLoss()
    criterion = MSELoss()
    test_loss = 0.0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            rgb_images = batch['rgb_image'].to(device)
            depth_images = batch['depth_image'].to(device)
            targets = batch['nutritional_values'].to(device)
            
            outputs = model(rgb_images, depth_images)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            # Store for metrics calculation
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    # Calculate metrics
    avg_test_loss = test_loss / len(test_loader)
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    
    # Calculate MAE for each component
    maes = torch.abs(all_outputs - all_targets).mean(dim=0)
    
    logging.info("\nTest Results:")
    logging.info(f"Average Test Loss: {avg_test_loss:.4f}")
    logging.info("MAE for each component:")
    logging.info(f"Mass: {maes[0]:.2f}g")
    logging.info(f"Fat: {maes[1]:.2f}g")
    logging.info(f"Carbs: {maes[2]:.2f}g")
    logging.info(f"Protein: {maes[3]:.2f}g")
    
    # Save predictions
    np.save('test_outputs.npy', all_outputs.numpy())
    np.save('test_targets.npy', all_targets.numpy())

if __name__ == '__main__':
    main()