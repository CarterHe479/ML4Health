import torch
import torch.optim as optim
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, loss, device=torch.device('cpu'), 
                 learning_rate=1e-4, weight_decay=1e-5, use_depth=True, work_dir='./'):
        """
        Args:
            model: The model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            device: Device to run the model on.
            learning_rate: Learning rate for the optimizer.
            weight_decay: Weight decay for the optimizer.
            use_depth: Whether to use depth images in training.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_depth = use_depth
        self.work_dir = work_dir
        
        self.loss = loss
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            factor=0.1, 
            patience=3
        )
        
        self.best_val_loss = float('inf')
        self.train_loss_history = []
        self.val_loss_history = []
        
        # Initialize plot
        plt.ion()  # Interactive mode on
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.train_line, = self.ax.plot([], [], 'b-', label='Training Loss', linewidth=2)
        self.val_line, = self.ax.plot([], [], 'r-', label='Validation Loss', linewidth=2)
        self.ax.set_xlabel('Epoch', fontsize=12)
        self.ax.set_ylabel('Loss', fontsize=12)
        self.ax.set_title('Live Training/Validation Loss', fontsize=14, pad=20)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.legend(fontsize=12)
        plt.tight_layout()

    def _update_plot(self):
        """Generate and save a fresh static plot each epoch"""
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(self.train_loss_history) + 1),
            self.train_loss_history,
            'b-',
            label='Training Loss',
            linewidth=2
        )
        plt.plot(
            range(1, len(self.val_loss_history) + 1),
            self.val_loss_history,
            'r-',
            label='Validation Loss',
            linewidth=2
        )
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training/Validation Loss', fontsize=14, pad=20)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.work_dir, 'loss_curves_final.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            rgb_images = batch['rgb_image'].to(self.device)                
            targets = batch['nutritional_values'].to(self.device)
            
            self.optimizer.zero_grad()
            if self.use_depth:
                depth_images = batch['depth_image'].to(self.device)
                outputs = self.model(rgb_images, depth_images)
            else:
                outputs = self.model(rgb_images)
            loss = self.loss(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(self.train_loader)
        self.train_loss_history.append(avg_train_loss)
        return avg_train_loss
    
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        
        progress_bar = tqdm(self.val_loader, desc="Validation")
        
        with torch.no_grad():
            for batch in progress_bar:
                rgb_images = batch['rgb_image'].to(self.device)
                
                targets = batch['nutritional_values'].to(self.device)
                
                if self.use_depth:
                    depth_images = batch['depth_image'].to(self.device)
                    outputs = self.model(rgb_images, depth_images)
                else:
                    outputs = self.model(rgb_images)
                loss = self.loss(outputs, targets)
                val_loss += loss.item()
                
                progress_bar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = val_loss / len(self.val_loader)
        self.val_loss_history.append(avg_val_loss)
        return avg_val_loss
    
    def train(self, num_epochs):
        try:
            for epoch in range(1, num_epochs + 1):
                train_loss = self.train_epoch(epoch)
                val_loss = self.validate()
                
                self.scheduler.step(val_loss)
                self._update_plot()
                
                logging.info(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )
                os.makedirs(self.work_dir, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.work_dir, 'latest.pth'))
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'val_loss': val_loss,
                        'epoch': epoch
                    }, os.path.join(self.work_dir, 'best_model.pth'))
                    logging.info(f"New best model saved with val loss: {val_loss:.4f}")
        
        finally:
            plt.ioff()  # Turn interactive mode off
            plt.close()  # Close the plot when done