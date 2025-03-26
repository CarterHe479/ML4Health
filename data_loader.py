import os
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

class Nutrition5KDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, val_ratio=0.2, random_seed=42):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Load split information
        with open('dataset_split.json', 'r') as f:
            split_info = json.load(f)
        
        # Split train into train/val if needed
        if split in ['train', 'val']:
            train_ids, val_ids = train_test_split(
                split_info['train_ids'],
                test_size=val_ratio,
                random_state=random_seed
            )
            self.dish_ids = train_ids if split == 'train' else val_ids
        else:  # test
            self.dish_ids = split_info['test_ids']
        
        # Load metadata
        self.metadata = self._load_metadata()
        self.samples = self._prepare_samples()
    
    def _load_metadata(self):
        """Load metadata from CSV files without headers, taking first 6 columns"""
        metadata_paths = [
            os.path.join(self.root_dir, 'metadata', 'dish_metadata_cafe1.csv'),
            os.path.join(self.root_dir, 'metadata', 'dish_metadata_cafe2.csv')
        ]
        
        dfs = []
        for path in metadata_paths:
            try:
                # Read CSV with no headers, only first 6 columns
                df = pd.read_csv(path, header=None, usecols=range(6))
                
                # Assign column names
                df.columns = [
                    'dish_id', 
                    'total_calories', 
                    'total_mass', 
                    'total_fat', 
                    'total_carb', 
                    'total_protein'
                ]
                
                # Convert numeric columns to float
                numeric_cols = df.columns[1:]  # all except dish_id
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                
                dfs.append(df)
                
            except Exception as e:
                print(f"Error reading {path}: {str(e)}")
                continue
        
        if not dfs:
            raise ValueError("No valid data found in metadata files")
        
        combined_df = pd.concat(dfs).drop_duplicates('dish_id')
        return combined_df.dropna()
  
    def _prepare_samples(self):
        """Prepare samples with image paths and nutritional values"""
        samples = []
        image_dir = os.path.join(self.root_dir, 'imagery', 'realsense_overhead')
        
        for dish_id in self.dish_ids:
            rgb_path = os.path.join(image_dir, dish_id, 'rgb.png')
            depth_path = os.path.join(image_dir, dish_id, 'depth_color.png')
            
            if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
                continue
                
            nutritional_values = self.metadata[self.metadata['dish_id'] == dish_id]
            if len(nutritional_values) == 0:
                continue
                
            nutritional_values = nutritional_values.iloc[0][[
                'total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein'
            ]].values.astype(float)
            
            samples.append({
                'dish_id': dish_id,
                'rgb_path': rgb_path,
                'depth_path': depth_path,
                'nutritional_values': nutritional_values
            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        rgb_image = Image.open(sample['rgb_path'])
        depth_image = Image.open(sample['depth_path'])
        
        if self.transform:
            rgb_image = self.transform(rgb_image)
            depth_image = self.transform(depth_image)
        
        nutritional_values = torch.tensor(sample['nutritional_values'], dtype=torch.float32)
        
        return {
            'dish_id': sample['dish_id'],
            'rgb_image': rgb_image,
            'depth_image': depth_image,
            'nutritional_values': nutritional_values
        }