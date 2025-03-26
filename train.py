import torch
import torch.optim as optim
from tqdm import tqdm
import logging
from loss import NutritionLoss

class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = NutritionLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        self.best_val_loss = float('inf')
        
    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                
                # Move data to device
                rgb_images = batch['rgb_image'].to(self.device)
                depth_images = batch['depth_image'].to(self.device)
                targets = batch['nutritional_values'].to(self.device)
                
                # Forward pass
                outputs = self.model(rgb_images, depth_images)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
        
        return train_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                rgb_images = batch['rgb_image'].to(self.device)
                depth_images = batch['depth_image'].to(self.device)
                targets = batch['nutritional_values'].to(self.device)
                
                outputs = self.model(rgb_images, depth_images)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
        
        return val_loss / len(self.val_loader)
    
    def train(self, num_epochs):
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            logging.info(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save last model
            torch.save(self.model.state_dict(), 'last_model.pth')
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                logging.info(f"New best model saved with val loss: {val_loss:.4f}")