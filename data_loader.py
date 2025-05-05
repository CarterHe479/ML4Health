import os
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import torch.nn.functional as F

class Nutrition5KDataset(Dataset):
    def __init__(self, root_dir, split='train', rgb_transform=None, depth_transform=None, val_ratio=0.2, random_seed=42):
        self.root_dir = root_dir
        self.split = split
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        
        # Load split info
        with open('dataset_split.json', 'r') as f:
            split_info = json.load(f)
        
        if split in ['train', 'val']:
            train_ids, val_ids = train_test_split(
                split_info['train_ids'],
                test_size=val_ratio,
                random_state=random_seed
            )
            self.dish_ids = train_ids if split == 'train' else val_ids
        else:
            self.dish_ids = split_info['test_ids']
        
        # Metadata
        self.metadata = self._load_metadata()
        self.samples = self._prepare_samples()
    
    def _load_metadata(self):
        metadata_paths = [
            os.path.join(self.root_dir, 'metadata', 'dish_metadata_cafe1.csv'),
            os.path.join(self.root_dir, 'metadata', 'dish_metadata_cafe2.csv')
        ]
        
        dfs = []
        for path in metadata_paths:
            df = pd.read_csv(path, header=None, usecols=range(6))
            df.columns = [
                'dish_id', 
                'total_calories', 
                'total_mass', 
                'total_fat', 
                'total_carb', 
                'total_protein'
            ]
            df[['total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein']] = \
                df[['total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein']].apply(pd.to_numeric, errors='coerce')
            dfs.append(df)
        
        combined_df = pd.concat(dfs).drop_duplicates('dish_id').dropna()
        return combined_df

    def _prepare_samples(self):
        samples = []
        image_dir = os.path.join(self.root_dir, 'imagery', 'realsense_overhead')
        
        for dish_id in self.dish_ids:
            rgb_path = os.path.join(image_dir, dish_id, 'rgb.png')
            depth_path = os.path.join(image_dir, dish_id, 'depth_color.png')
            side_paths = [
                os.path.join(image_dir, dish_id, f'camera_{c}.h264') for c in ['A', 'B', 'C', 'D']
            ]
            
            if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
                continue
                
            meta_row = self.metadata[self.metadata['dish_id'] == dish_id]
            if len(meta_row) == 0:
                continue
                
            nutrition = meta_row.iloc[0][['total_fat', 'total_carb', 'total_protein', 'total_calories']].values.astype(float)
            
            samples.append({
                'dish_id': dish_id,
                'rgb_path': rgb_path,
                'depth_path': depth_path,
                'side_paths': side_paths,
                'nutrition': nutrition
            })
        
        return samples
    
    def _load_side_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            # blank image if missing
            return torch.zeros(3, 480, 640)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return frame

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load RGB
        rgb_image = Image.open(sample['rgb_path']).convert('RGB')
        rgb_tensor = torch.from_numpy(np.array(rgb_image)).permute(2, 0, 1).float() / 255.0

        # Load one side frame (you can change to use more if needed)
        side_frame = self._load_side_frame(sample['side_paths'][0])  # [3, H, W]

        # Concat RGB + side
        rgb_side = torch.cat([rgb_tensor, side_frame], dim=0)  # [6, H, W]

        # ✅ Instead of self.rgb_transform:
        rgb_side = F.interpolate(rgb_side.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)

        mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225]).view(-1, 1, 1)
        rgb_side = (rgb_side - mean) / std

        # Depth image
        depth_image = Image.open(sample['depth_path']).convert('L')
        if self.depth_transform:
            depth_image = self.depth_transform(depth_image)
        else:
            depth_image = torch.from_numpy(np.array(depth_image)).unsqueeze(0).float() / 255.0
        
        nutrition_tensor = torch.tensor(sample['nutrition'], dtype=torch.float32)
        
        return {
            'dish_id': sample['dish_id'],
            'rgb_side': rgb_side,
            'depth_image': depth_image,
            'nutrition': nutrition_tensor
        }
