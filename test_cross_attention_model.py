import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader

# Import your modules
from data_loader import Nutrition5KDataset
from cross_attention_model import CrossAttentionNutritionModel  # 你的cross attention模型
from utils import setup_logging, get_device

def load_model(model_path, device):
    """Load the trained cross-attention model."""
    model = CrossAttentionNutritionModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']} with val loss {checkpoint['val_loss']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights (no epoch/loss info available)")
    return model

def evaluate_model(model, data_loader, device):
    model.eval()
    all_targets = {'mass': [], 'fat': [], 'carbs': [], 'protein': []}
    all_predictions = {'mass': [], 'fat': [], 'carbs': [], 'protein': []}
    all_dish_ids = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            rgb_side = batch['rgb_side'].to(device)
            depth_images = batch['depth_image'].to(device)
            targets = batch['nutrition'].to(device)
            dish_ids = batch['dish_id']

            outputs = model(rgb_side, depth_images)
            all_dish_ids.extend(dish_ids)

            all_targets['mass'].extend(targets[:, 0].cpu().numpy())
            all_targets['fat'].extend(targets[:, 1].cpu().numpy())
            all_targets['carbs'].extend(targets[:, 2].cpu().numpy())
            all_targets['protein'].extend(targets[:, 3].cpu().numpy())

            predictions = outputs.cpu().numpy()
            all_predictions['mass'].extend(predictions[:, 0])
            all_predictions['fat'].extend(predictions[:, 1])
            all_predictions['carbs'].extend(predictions[:, 2])
            all_predictions['protein'].extend(predictions[:, 3])

    metrics = {}
    nutrients = ['mass', 'fat', 'carbs', 'protein']
    for nutrient in nutrients:
        y_true = np.array(all_targets[nutrient])
        y_pred = np.array(all_predictions[nutrient])
        metrics[nutrient] = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mean_true': np.mean(y_true),
            'mean_pred': np.mean(y_pred),
            'correlation': np.corrcoef(y_true, y_pred)[0, 1],
            'nrmse': (np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(y_true) * 100) if np.mean(y_true) > 0 else float('inf')
        }
    results_df = pd.DataFrame({'dish_id': all_dish_ids})
    for nutrient in nutrients:
        results_df[f'{nutrient}_true'] = all_targets[nutrient]
        results_df[f'{nutrient}_pred'] = all_predictions[nutrient]
        results_df[f'{nutrient}_error'] = results_df[f'{nutrient}_pred'] - results_df[f'{nutrient}_true']
        results_df[f'{nutrient}_abs_error'] = abs(results_df[f'{nutrient}_error'])
    return metrics, results_df, {'predictions': all_predictions, 'targets': all_targets}

def main():
    setup_logging()
    device = get_device()

    results_dir = 'test_results_cross_attention'
    os.makedirs(results_dir, exist_ok=True)
    vis_dir = os.path.join(results_dir, 'visualizations')
    analysis_dir = os.path.join(results_dir, 'error_analysis')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    model_path = 'cross_attention_model/best_model.pth'

    # No transforms passed, matching your current logic
    logging.info("Loading test dataset...")
    root_dir = './nutrition5k_dataset'
    test_dataset = Nutrition5KDataset(
        root_dir=root_dir,
        split='test',
        rgb_transform=None,
        depth_transform=None
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    logging.info(f"Loading model from {model_path}...")
    model = load_model(model_path, device)

    logging.info("Evaluating model...")
    metrics, results_df, prediction_data = evaluate_model(model, test_loader, device)

    logging.info("\n===== Model Performance Metrics =====")
    for nutrient, metric in metrics.items():
        logging.info(f"\n{nutrient.upper()} Metrics:")
        logging.info(f"  MSE: {metric['mse']:.4f}")
        logging.info(f"  RMSE: {metric['rmse']:.4f} g")
        logging.info(f"  MAE: {metric['mae']:.4f} g")
        logging.info(f"  R²: {metric['r2']:.4f}")
        logging.info(f"  Correlation: {metric['correlation']:.4f}")
        logging.info(f"  NRMSE: {metric['nrmse']:.2f}% of mean value")
        logging.info(f"  Ground Truth Mean: {metric['mean_true']:.2f} g")
        logging.info(f"  Prediction Mean: {metric['mean_pred']:.2f} g")

    results_df.to_csv(os.path.join(results_dir, 'all_predictions.csv'), index=False)

    # (Optionally you can copy `visualize_predictions` and `analyze_worst_predictions` from the previous script if needed)

    logging.info(f"Evaluation complete. Results saved to {results_dir}")

if __name__ == '__main__':
    main()
