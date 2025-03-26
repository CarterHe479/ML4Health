import logging
import torch

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler('testing.log'),
            logging.StreamHandler()
        ]
    )
def get_device():
    device = torch.accelerator.current_accelerator()
    print(f"Using device: {device}")
    return device
