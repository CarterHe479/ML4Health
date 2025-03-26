import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import Nutrition5KDataset
from cnn_model import NutritionCNN
from loss import NutritionLoss
import logging

def test_model():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler('testing.log'),
            logging.StreamHandler()
        ]
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = NutritionCNN().to(device)
    model.load_state_dict(torch.load('last_model.pth')) # It should be best_model.pth
    model.eval()
    
    # Load test data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = Nutrition5KDataset(
        root_dir='/Users/liwuchen/Documents/nutrition5k_dataset',
        split='test',
        transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Test the model
    criterion = NutritionLoss()
    test_loss = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            rgb_images = batch['rgb_image'].to(device)
            depth_images = batch['depth_image'].to(device)
            targets = batch['nutritional_values'].to(device)
            
            outputs = model(rgb_images, depth_images)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    logging.info(f"Test Loss: {avg_test_loss:.4f}")

if __name__ == '__main__':
    test_model()