"""
Configuration file for UNet training
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data paths
    train_images: str = "data/train/images"
    train_masks: str = "data/train/masks"
    val_images: str = "data/val/images"
    val_masks: str = "data/val/masks"
    
    # Model parameters
    input_channels: int = 3
    num_classes: int = 1
    bilinear: bool = False
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    loss_function: str = "dice_bce"
    threshold: float = 0.5
    
    # Data parameters
    image_size: int = 512
    num_workers: int = 4
    
    # Output parameters
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    output_dir: str = "outputs"
    save_freq: int = 10
    plot_freq: int = 20
    
    # Resume training
    resume: Optional[str] = None


@dataclass
class InferenceConfig:
    """Inference configuration"""
    # Model parameters
    checkpoint_path: str = "checkpoints/best_model.pth"
    input_channels: int = 3
    num_classes: int = 1
    bilinear: bool = False
    
    # Processing parameters
    image_size: int = 512
    threshold: float = 0.5
    
    # Output parameters
    output_dir: str = "predictions"


# Default configurations
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_INFERENCE_CONFIG = InferenceConfig()


def create_experiment_config(experiment_name: str, **kwargs) -> TrainingConfig:
    """Create a configuration for a specific experiment"""
    config = TrainingConfig()
    
    # Update with experiment-specific parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Update output directories with experiment name
    config.checkpoint_dir = f"checkpoints/{experiment_name}"
    config.log_dir = f"logs/{experiment_name}"
    config.output_dir = f"outputs/{experiment_name}"
    
    return config


# Example experiment configurations
EXPERIMENT_CONFIGS = {
    "medical_segmentation": TrainingConfig(
        epochs=150,
        batch_size=4,
        learning_rate=5e-5,
        loss_function="dice_bce",
        image_size=256,
        bilinear=True
    ),
    
    "satellite_imagery": TrainingConfig(
        epochs=200,
        batch_size=16,
        learning_rate=1e-4,
        loss_function="focal",
        image_size=1024,
        bilinear=False
    ),
    
    "cell_segmentation": TrainingConfig(
        epochs=100,
        batch_size=8,
        learning_rate=1e-4,
        loss_function="dice",
        image_size=512,
        bilinear=False
    )
}


def save_config(config: TrainingConfig, save_path: str):
    """Save configuration to file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        for key, value in config.__dict__.items():
            f.write(f"{key} = {value}\n")


def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from file"""
    config = TrainingConfig()
    
    with open(config_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Convert value to appropriate type
                if hasattr(config, key):
                    attr_type = type(getattr(config, key))
                    if attr_type == bool:
                        setattr(config, key, value.lower() == 'true')
                    elif attr_type == int:
                        setattr(config, key, int(value))
                    elif attr_type == float:
                        setattr(config, key, float(value))
                    else:
                        setattr(config, key, value)
    
    return config 