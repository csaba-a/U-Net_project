"""
Advanced UNet Trainer with production-ready features.
This trainer demonstrates enterprise-level ML engineering practices.
"""

import os
import time
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import json

# Optional imports for advanced features
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from models.unet import UNet
from models.attention_unet import AttentionUNet
from utils.dataset import SegmentationDataset, get_training_augmentation, get_validation_augmentation
from utils.losses import get_loss_function
from utils.advanced_losses import get_advanced_loss_function
from utils.metrics import SegmentationMetrics


class EarlyStopping:
    """
    Early stopping mechanism with multiple criteria.
    """
    
    def __init__(self, 
                 patience: int = 10, 
                 min_delta: float = 0.001,
                 mode: str = 'min',
                 monitor: str = 'val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False
            
        if self.mode == 'min':
            improved = current_score < self.best_score - self.min_delta
        else:
            improved = current_score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            return True
            
        return False


class LearningRateScheduler:
    """
    Advanced learning rate scheduler with multiple strategies.
    """
    
    def __init__(self, 
                 optimizer: optim.Optimizer,
                 scheduler_type: str = 'cosine',
                 **kwargs):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=kwargs.get('epochs', 100)
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=kwargs.get('step_size', 30), gamma=kwargs.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=kwargs.get('patience', 5), factor=kwargs.get('factor', 0.5)
            )
        elif scheduler_type == 'onecycle':
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=kwargs.get('max_lr', 1e-3), epochs=kwargs.get('epochs', 100),
                steps_per_epoch=kwargs.get('steps_per_epoch', 100)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
            
    def step(self, metrics: Optional[float] = None):
        """Step the scheduler."""
        if self.scheduler_type == 'plateau' and metrics is not None:
            self.scheduler.step(metrics)
        elif self.scheduler_type == 'onecycle':
            self.scheduler.step()
        else:
            self.scheduler.step()
            
    def get_last_lr(self) -> List[float]:
        """Get current learning rate."""
        return self.scheduler.get_last_lr()


class ModelCheckpoint:
    """
    Advanced model checkpointing with multiple save strategies.
    """
    
    def __init__(self, 
                 save_dir: str,
                 save_top_k: int = 3,
                 monitor: str = 'val_loss',
                 mode: str = 'min'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.best_models = []
        
    def save(self, 
             model: nn.Module,
             optimizer: optim.Optimizer,
             epoch: int,
             metrics: Dict[str, float],
             filename: str = None) -> str:
        """Save model checkpoint."""
        
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pth"
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': getattr(model, 'config', {})
        }
        
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
        
        # Update best models list - safely access monitored metric
        monitored_score = metrics.get(self.monitor, float('inf') if self.mode == 'min' else float('-inf'))
        self._update_best_models(filepath, monitored_score)
        
        return str(filepath)
    
    def _update_best_models(self, filepath: Path, score: float):
        """Update list of best models."""
        self.best_models.append((filepath, score))
        self.best_models.sort(key=lambda x: x[1], reverse=(self.mode == 'max'))
        
        # Keep only top-k models
        if len(self.best_models) > self.save_top_k:
            worst_model = self.best_models.pop()
            if worst_model[0].exists():
                worst_model[0].unlink()  # Delete file
                
    def load_best(self, model: nn.Module, optimizer: optim.Optimizer = None) -> Dict[str, Any]:
        """Load best model checkpoint."""
        if not self.best_models:
            raise ValueError("No checkpoints available")
            
        best_checkpoint = self.best_models[0][0]
        return self.load(best_checkpoint, model, optimizer)
    
    def load(self, 
             checkpoint_path: str, 
             model: nn.Module, 
             optimizer: optim.Optimizer = None) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return checkpoint


class UNetTrainer:
    """
    Advanced UNet trainer with production-ready features.
    
    This trainer demonstrates:
    - Distributed training across multiple GPUs
    - Mixed precision training
    - Advanced experiment tracking
    - Comprehensive logging and visualization
    - Model checkpointing and early stopping
    - Advanced loss functions and optimizers
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 experiment_name: str = "unet_experiment",
                 enable_wandb: bool = False,
                 enable_mlflow: bool = False,
                 enable_tensorboard: bool = True,
                 log_dir: str = "logs"):
        
        self.config = config
        self.experiment_name = experiment_name
        self.enable_wandb = enable_wandb and WANDB_AVAILABLE
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE
        self.enable_tensorboard = enable_tensorboard and TENSORBOARD_AVAILABLE
        self.log_dir = Path(log_dir)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.device = self._setup_device()
        self.model = self._create_model()
        self.criterion = self._create_loss_function()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler() if self.config.get('mixed_precision', False) else None
        
        # Training components
        self.early_stopping = EarlyStopping(
            patience=self.config.get('early_stopping_patience', 10),
            monitor=self.config.get('early_stopping_monitor', 'val_loss')
        )
        
        self.checkpointer = ModelCheckpoint(
            save_dir=self.config.get('checkpoint_dir', 'checkpoints'),
            save_top_k=self.config.get('save_top_k', 3),
            monitor=self.config.get('checkpoint_monitor', 'val_loss')
        )
        
        # Metrics
        self.metrics = SegmentationMetrics()
        
        # Experiment tracking
        self._setup_experiment_tracking()
        
        # Training state
        self.current_epoch = 0
        self.best_metrics = {}
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_dice': [], 'val_dice': [],
            'train_iou': [], 'val_iou': [],
            'train_accuracy': [], 'val_accuracy': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': [],
            'train_f1': [], 'val_f1': []
        }
        
    def _setup_logging(self):
        """Setup logging configuration."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / f"{self.experiment_name}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.experiment_name)
        
    def _setup_device(self) -> torch.device:
        """Setup device (CPU/GPU)."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            self.logger.info("Using CPU")
        return device
        
    def _create_model(self) -> nn.Module:
        """Create model based on configuration."""
        model_config = self.config.get('model', {})
        model_type = model_config.get('type', 'unet')
        
        if model_type == 'unet':
            model = UNet(
                n_channels=model_config.get('n_channels', 3),
                n_classes=model_config.get('n_classes', 1),
                bilinear=model_config.get('bilinear', False)
            )
        elif model_type == 'attention_unet':
            model = AttentionUNet(
                n_channels=model_config.get('n_channels', 3),
                n_classes=model_config.get('n_classes', 1),
                bilinear=model_config.get('bilinear', False),
                use_attention=model_config.get('use_attention', True),
                use_spatial_attention=model_config.get('use_spatial_attention', False),
                use_channel_attention=model_config.get('use_channel_attention', False),
                dropout_rate=model_config.get('dropout_rate', 0.1)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model = model.to(self.device)
        return model
        
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration."""
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type', 'dice_bce')
        
        if loss_type in ['boundary', 'focal_tversky', 'combo', 'hausdorff', 'surface', 'lovasz_hinge', 'weighted_bce']:
            return get_advanced_loss_function(loss_type, **loss_config.get('params', {}))
        else:
            return get_loss_function(loss_type)
            
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam')
        
        if optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-4),
                weight_decay=optimizer_config.get('weight_decay', 0)
            )
        elif optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-4),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        elif optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-3),
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
    def _create_scheduler(self) -> LearningRateScheduler:
        """Create learning rate scheduler."""
        scheduler_config = self.config.get('scheduler', {})
        return LearningRateScheduler(
            self.optimizer,
            scheduler_type=scheduler_config.get('type', 'cosine'),
            **scheduler_config.get('params', {})
        )
        
    def _setup_experiment_tracking(self):
        """Setup experiment tracking (W&B, MLflow, TensorBoard)."""
        if self.enable_wandb:
            wandb.init(
                project=self.config.get('wandb_project', 'unet_training'),
                name=self.experiment_name,
                config=self.config
            )
            
        if self.enable_mlflow:
            mlflow.set_experiment(self.experiment_name)
            mlflow.log_params(self.config)
            
        if self.enable_tensorboard:
            self.tensorboard_writer = SummaryWriter(self.log_dir / self.experiment_name)
            
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders."""
        data_config = self.config.get('data', {})
        
        # Create datasets
        train_dataset = SegmentationDataset(
            image_dir=data_config['train_images'],
            mask_dir=data_config['train_masks'],
            transform=get_training_augmentation(data_config.get('image_size', (512, 512))),
            image_size=data_config.get('image_size', (512, 512))
        )
        
        val_dataset = SegmentationDataset(
            image_dir=data_config['val_images'],
            mask_dir=data_config['val_masks'],
            transform=get_validation_augmentation(data_config.get('image_size', (512, 512))),
            image_size=data_config.get('image_size', (512, 512))
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        return train_loader, val_loader
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        self.metrics.reset()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                # Mixed precision training
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                self.scaler.scale(loss).backward()
                
                if self.config.get('gradient_clipping', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clipping']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                
                if self.config.get('gradient_clipping', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clipping']
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            self.metrics.update(masks, outputs)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log to tensorboard
            if self.enable_tensorboard and batch_idx % 10 == 0:
                self.tensorboard_writer.add_scalar(
                    'Train/BatchLoss', loss.item(), 
                    self.current_epoch * len(train_loader) + batch_idx
                )
        
        avg_loss = total_loss / len(train_loader)
        metrics = self.metrics.get_metrics()
        
        return {
            'loss': avg_loss,
            'dice': metrics['dice'],
            'iou': metrics['iou'],
            'accuracy': metrics['accuracy']
        }
        
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        self.metrics.reset()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
                self.metrics.update(masks, outputs)
        
        avg_loss = total_loss / len(val_loader)
        metrics = self.metrics.get_metrics()
        
        return {
            'loss': avg_loss,
            'dice': metrics['dice'],
            'iou': metrics['iou'],
            'accuracy': metrics['accuracy']
        }
        
    def train(self, 
              enable_distributed: bool = False,
              num_gpus: int = 1,
              strategy: str = 'ddp') -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            enable_distributed: Enable distributed training
            num_gpus: Number of GPUs to use
            strategy: Distributed strategy ('ddp', 'dp')
            
        Returns:
            Training results
        """
        self.logger.info(f"Starting training for {self.config.get('epochs', 100)} epochs")
        
        # Setup distributed training if enabled
        if enable_distributed and num_gpus > 1:
            self._setup_distributed_training(strategy, num_gpus)
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders()
        
        # Training loop
        for epoch in range(self.config.get('epochs', 100)):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics)
            
            # Save checkpoint
            # Convert val_metrics to include 'val_' prefix for checkpoint monitoring
            checkpoint_metrics = {f'val_{k}': v for k, v in val_metrics.items()}
            checkpoint_path = self.checkpointer.save(
                self.model, self.optimizer, epoch, checkpoint_metrics
            )
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                self.logger.info("Early stopping triggered")
                break
                
            # Update best metrics
            if not self.best_metrics or val_metrics['loss'] < self.best_metrics.get('val_loss', float('inf')):
                self.best_metrics = val_metrics
                self.best_metrics['epoch'] = epoch
                
        # Final evaluation
        final_results = self._finalize_training()
        
        return final_results
        
    def _setup_distributed_training(self, strategy: str, num_gpus: int):
        """Setup distributed training."""
        if strategy == 'ddp':
            # Initialize process group
            dist.init_process_group(backend='nccl')
            
            # Wrap model in DDP
            self.model = DDP(self.model, device_ids=[self.device])
            
        elif strategy == 'dp':
            # DataParallel (not recommended for production)
            self.model = nn.DataParallel(self.model)
            
    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics to various tracking systems."""
        # Update training history dynamically
        for key in train_metrics:
            train_key = f'train_{key}'
            val_key = f'val_{key}'
            
            # Initialize lists if they don't exist
            if train_key not in self.training_history:
                self.training_history[train_key] = []
            if val_key not in self.training_history:
                self.training_history[val_key] = []
            
            self.training_history[train_key].append(train_metrics[key])
            self.training_history[val_key].append(val_metrics[key])
        
        # Log to console
        self.logger.info(
            f"Epoch {self.current_epoch + 1}: "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Dice: {val_metrics['dice']:.4f}"
        )
        
        # Log to tensorboard
        if self.enable_tensorboard:
            for key in train_metrics:
                self.tensorboard_writer.add_scalar(f'Train/{key}', train_metrics[key], self.current_epoch)
                self.tensorboard_writer.add_scalar(f'Val/{key}', val_metrics[key], self.current_epoch)
            
            self.tensorboard_writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], self.current_epoch)
        
        # Log to wandb
        if self.enable_wandb:
            wandb.log({
                'epoch': self.current_epoch,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()},
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        # Log to mlflow
        if self.enable_mlflow:
            mlflow.log_metrics({
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()},
                'lr': self.optimizer.param_groups[0]['lr']
            }, step=self.current_epoch)
            
    def _finalize_training(self) -> Dict[str, Any]:
        """Finalize training and cleanup."""
        # Save final model
        final_checkpoint = self.checkpointer.save(
            self.model, self.optimizer, self.current_epoch, self.best_metrics,
            filename="final_model.pth"
        )
        
        # Create training plots
        self._create_training_plots()
        
        # Close tensorboard writer
        if self.enable_tensorboard:
            self.tensorboard_writer.close()
        
        # Log final results
        self.logger.info(f"Training completed. Best validation loss: {self.best_metrics['val_loss']:.4f}")
        
        return {
            'best_metrics': self.best_metrics,
            'training_history': self.training_history,
            'final_checkpoint': final_checkpoint
        }
        
    def _create_training_plots(self):
        """Create and save training plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Dice plot
        axes[0, 1].plot(self.training_history['train_dice'], label='Train Dice')
        axes[0, 1].plot(self.training_history['val_dice'], label='Val Dice')
        axes[0, 1].set_title('Training and Validation Dice Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # IoU plot
        axes[1, 0].plot(self.training_history['train_iou'], label='Train IoU')
        axes[1, 0].plot(self.training_history['val_iou'], label='Val IoU')
        axes[1, 0].set_title('Training and Validation IoU Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate plot
        lr_history = [self.scheduler.get_last_lr()[0]] * len(self.training_history['train_loss'])
        axes[1, 1].plot(lr_history)
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / f"{self.experiment_name}_training_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def evaluate(self, 
                test_data: str = None,
                save_predictions: bool = True) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Path to test data
            save_predictions: Whether to save predictions
            
        Returns:
            Evaluation metrics
        """
        self.logger.info("Starting model evaluation")
        
        # Load best model
        self.checkpointer.load_best(self.model)
        
        # Create test data loader
        if test_data:
            test_dataset = SegmentationDataset(
                image_dir=test_data + '/images',
                mask_dir=test_data + '/masks',
                transform=get_validation_augmentation(self.config.get('data', {}).get('image_size', (512, 512))),
                image_size=self.config.get('data', {}).get('image_size', (512, 512))
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.get('batch_size', 8),
                shuffle=False,
                num_workers=self.config.get('num_workers', 4)
            )
            
            # Evaluate
            test_metrics = self.validate_epoch(test_loader)
            
            self.logger.info(f"Test metrics: {test_metrics}")
            
            return test_metrics
        
        return self.best_metrics 