import os
import json
import random
from sklearn.model_selection import train_test_split

def save_dataset_split(root_dir, n_samples=1000, test_ratio=0.2, random_seed=42):
    """Save train/test splits to file with sampling"""
    image_dir = os.path.join(root_dir, 'imagery', 'realsense_overhead')
    all_dish_ids = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    
    # Random sample n_samples dish IDs
    random.seed(random_seed)
    sampled_ids = random.sample(all_dish_ids, min(n_samples, len(all_dish_ids)))
    
    # Split into train and test
    train_ids, test_ids = train_test_split(
        sampled_ids, 
        test_size=test_ratio, 
        random_state=random_seed
    )
    
    # Save to file
    split_info = {
        'train_ids': train_ids,
        'test_ids': test_ids,
    }
    
    with open('dataset_split.json', 'w') as f:
        json.dump(split_info, f)
    
    print(f"Saved {len(train_ids)} train and {len(test_ids)} test samples")

if __name__ == '__main__':
    root_dir = '/Users/liwuchen/Documents/nutrition5k_dataset'
    save_dataset_split(root_dir, n_samples=1000, test_ratio=0.2)