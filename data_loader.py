import os
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

class Nutrition5KDataset(Dataset):
    def __init__(self, root_dir, split='train', rgb_transform=None, depth_transform=None, val_ratio=0.2, random_seed=42):
        self.root_dir = root_dir
        self.split = split
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

        # Side video config
        self.side_camera_names = ["camera_A.h264", "camera_B.h264", "camera_C.h264", "camera_D.h264"]
        self.side_frame_cache = {}  # {video_path: frame_tensor}

        # Load split information
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

        self.metadata = self._load_metadata()
        self.samples = self._prepare_samples()

    def _load_metadata(self):
        metadata_paths = [
            os.path.join(self.root_dir, 'metadata', 'dish_metadata_cafe1.csv'),
            os.path.join(self.root_dir, 'metadata', 'dish_metadata_cafe2.csv')
        ]
        df_list = []
        for path in metadata_paths:
            df = pd.read_csv(path, header=None)
            df_list.append(df.iloc[:, :6])
        metadata = pd.concat(df_list, ignore_index=True)
        metadata.columns = ['id', 'fat_g', 'carb_g', 'protein_g', 'mass_g', 'kcal']
        return metadata

    def _prepare_samples(self):
        samples = []
        for dish_id in self.dish_ids:
            rgb_path = os.path.join(self.root_dir, "imagery", "realsense_overhead", dish_id, "rgb.png")
            depth_path = os.path.join(self.root_dir, "imagery", "realsense_overhead", dish_id, "depth_color.png")
            samples.append({"dish_id": dish_id, "rgb": rgb_path, "depth": depth_path})
        return samples

    def _load_side_frame(self, video_path):
        if video_path in self.side_frame_cache:
            return self.side_frame_cache[video_path]
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read frame from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        self.side_frame_cache[video_path] = frame
        return frame

    def __getitem__(self, idx):
        sample = self.samples[idx]
        rgb_path = sample["rgb"]
        depth_path = sample["depth"]
        dish_id = sample["dish_id"]

        rgb_image = Image.open(rgb_path).convert("RGB")
        depth_image = Image.open(depth_path).convert("L")

        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image)
        else:
            rgb_image = torch.from_numpy(np.array(rgb_image)).permute(2, 0, 1).float() / 255.0

        if self.depth_transform:
            depth_image = self.depth_transform(depth_image)
        else:
            depth_image = torch.from_numpy(np.array(depth_image)).unsqueeze(0).float() / 255.0

        # Load side camera frames
        dish_folder = os.path.join(self.root_dir, "imagery", "realsense_overhead", dish_id)
        side_images = []
        for cam_name in self.side_camera_names:
            video_path = os.path.join(dish_folder, cam_name)
            side_frame = self._load_side_frame(video_path)
            side_images.append(side_frame)
        side_images = torch.stack(side_images)  # [4, 3, 480, 640]

        # 选择一个 side 图像（例如 camera_A）
        side_image = side_images[0]  # [3, 480, 640]

        # 拼接 RGB + side => [6, 480, 640]
        rgb_side_image = torch.cat([rgb_image, side_image], dim=0)

        # 获取标签
        meta_row = self.metadata[self.metadata['id'] == int(dish_id)]
        if meta_row.empty:
            raise ValueError(f"Dish ID {dish_id} not found in metadata.")
        nutrition = meta_row.iloc[0][['fat_g', 'carb_g', 'protein_g', 'kcal']].values.astype(np.float32)
        nutrition = torch.tensor(nutrition)

        return {
            "rgb_side": rgb_side_image,   # [6, 480, 640]
            "depth": depth_image,         # [1, 480, 640]
            "label": nutrition,           # [4]
            "dish_id": dish_id
        }


    def __len__(self):
        return len(self.samples)


# For testing the dataset class
# from torchvision import transforms

# # Define basic transforms if needed
# rgb_transform = transforms.Compose([
#     transforms.ToTensor(),
# ])

# depth_transform = transforms.Compose([
#     transforms.ToTensor(),
# ])

# # Initialize dataset
# dataset = Nutrition5KDataset(
#     root_dir="~/Documents/nutrition5k_dataset",  # adjust if needed
#     split="train",
#     rgb_transform=rgb_transform,
#     depth_transform=depth_transform
# )

# # Load one sample
# sample = dataset[0]

# print("RGB+Side shape:", sample["rgb_side"].shape)  # [6, 480, 640]
# print("Depth shape:", sample["depth"].shape)        # [1, 480, 640]
# print("Label:", sample["label"])                    # [fat, carb, protein, kcal]
# print("Dish ID:", sample["dish_id"])

