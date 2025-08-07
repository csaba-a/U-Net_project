"""
Advanced visualization tools for UNet training and model interpretability.
This module demonstrates understanding of visualization best practices and model interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from PIL import Image
import cv2
from typing import List, Tuple, Optional, Dict, Any
import os
from pathlib import Path


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for model interpretability.
    This implementation demonstrates understanding of attention mechanisms and model interpretability.
    """
    
    def __init__(self, model: nn.Module, target_layer: str = None):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks to capture gradients and activations."""
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        # Find target layer
        if self.target_layer is None:
            # Default to last convolutional layer
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    self.target_layer = name
                    
        # Register hooks
        target_module = dict(self.model.named_modules())[self.target_layer]
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)
        
    def generate_cam(self, input_image: torch.Tensor, target_class: int = 0) -> np.ndarray:
        """
        Generate Grad-CAM for the given input image.
        
        Args:
            input_image: Input image tensor (B, C, H, W)
            target_class: Target class for visualization
            
        Returns:
            Grad-CAM heatmap
        """
        # Forward pass
        output = self.model(input_image)
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()
        
        # Calculate weights
        weights = torch.mean(gradients, dim=[2, 3])
        
        # Generate CAM
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
            
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.numpy()
        
    def visualize(self, 
                 input_image: torch.Tensor, 
                 target_class: int = 0,
                 save_path: str = None) -> plt.Figure:
        """
        Visualize Grad-CAM with original image.
        
        Args:
            input_image: Input image tensor
            target_class: Target class
            save_path: Path to save visualization
            
        Returns:
            Matplotlib figure
        """
        # Generate CAM
        cam = self.generate_cam(input_image, target_class)
        
        # Convert input to numpy
        img = input_image.squeeze().permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Grad-CAM heatmap
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Overlay
        overlay = img.copy()
        cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
        cam_colored = plt.cm.jet(cam_resized)[:, :, :3]
        overlay = 0.7 * overlay + 0.3 * cam_colored
        overlay = np.clip(overlay, 0, 1)
        
        axes[2].imshow(overlay)
        axes[2].set_title('Grad-CAM Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


class AttentionVisualizer:
    """
    Visualize attention mechanisms in Attention UNet.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_maps = {}
        
    def capture_attention_maps(self, input_image: torch.Tensor):
        """Capture attention maps during forward pass."""
        # This would need to be implemented based on the specific attention mechanism
        # For now, we'll create a placeholder
        pass
        
    def visualize_attention(self, 
                          input_image: torch.Tensor,
                          save_path: str = None) -> plt.Figure:
        """Visualize attention maps."""
        # Placeholder implementation
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.text(0.5, 0.5, 'Attention Visualization\n(Implementation depends on model architecture)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


class TrainingVisualizer:
    """
    Comprehensive training visualization tools.
    """
    
    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_training_curves(self, 
                           history: Dict[str, List[float]],
                           save_path: str = None) -> plt.Figure:
        """
        Plot comprehensive training curves.
        
        Args:
            history: Dictionary containing training history
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss curves
        if 'train_loss' in history and 'val_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
            axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
        # Dice score curves
        if 'train_dice' in history and 'val_dice' in history:
            axes[0, 1].plot(history['train_dice'], label='Train Dice', linewidth=2)
            axes[0, 1].plot(history['val_dice'], label='Val Dice', linewidth=2)
            axes[0, 1].set_title('Training and Validation Dice Score')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Dice Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
        # IoU score curves
        if 'train_iou' in history and 'val_iou' in history:
            axes[0, 2].plot(history['train_iou'], label='Train IoU', linewidth=2)
            axes[0, 2].plot(history['val_iou'], label='Val IoU', linewidth=2)
            axes[0, 2].set_title('Training and Validation IoU Score')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('IoU Score')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
        # Learning rate curve
        if 'lr' in history:
            axes[1, 0].plot(history['lr'], linewidth=2, color='purple')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)
            
        # Accuracy curves
        if 'train_accuracy' in history and 'val_accuracy' in history:
            axes[1, 1].plot(history['train_accuracy'], label='Train Accuracy', linewidth=2)
            axes[1, 1].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
            axes[1, 1].set_title('Training and Validation Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
        # Loss distribution
        if 'train_loss' in history:
            axes[1, 2].hist(history['train_loss'], bins=30, alpha=0.7, label='Train Loss')
            if 'val_loss' in history:
                axes[1, 2].hist(history['val_loss'], bins=30, alpha=0.7, label='Val Loss')
            axes[1, 2].set_title('Loss Distribution')
            axes[1, 2].set_xlabel('Loss')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_confusion_matrix(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            save_path: str = None) -> plt.Figure:
        """
        Plot confusion matrix for segmentation results.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_metrics_comparison(self, 
                              metrics_dict: Dict[str, Dict[str, float]],
                              save_path: str = None) -> plt.Figure:
        """
        Plot comparison of different models or configurations.
        
        Args:
            metrics_dict: Dictionary of metrics for different models
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract metrics
        models = list(metrics_dict.keys())
        dice_scores = [metrics_dict[model].get('dice', 0) for model in models]
        iou_scores = [metrics_dict[model].get('iou', 0) for model in models]
        
        # Bar plot
        x = np.arange(len(models))
        width = 0.35
        
        axes[0].bar(x - width/2, dice_scores, width, label='Dice Score', alpha=0.8)
        axes[0].bar(x + width/2, iou_scores, width, label='IoU Score', alpha=0.8)
        axes[0].set_xlabel('Models')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Model Performance Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Radar chart
        categories = ['Dice', 'IoU', 'Accuracy', 'Precision', 'Recall', 'F1']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Get metrics for first model (example)
        first_model = list(metrics_dict.values())[0]
        values = [
            first_model.get('dice', 0),
            first_model.get('iou', 0),
            first_model.get('accuracy', 0),
            first_model.get('precision', 0),
            first_model.get('recall', 0),
            first_model.get('f1', 0)
        ]
        values += values[:1]  # Complete the circle
        
        axes[1].plot(angles, values, 'o-', linewidth=2, label=models[0])
        axes[1].fill(angles, values, alpha=0.25)
        axes[1].set_xticks(angles[:-1])
        axes[1].set_xticklabels(categories)
        axes[1].set_ylim(0, 1)
        axes[1].set_title('Metrics Radar Chart')
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


class SegmentationVisualizer:
    """
    Advanced segmentation visualization tools.
    """
    
    def __init__(self, save_dir: str = "segmentations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def visualize_prediction(self, 
                           image: np.ndarray,
                           mask: np.ndarray,
                           prediction: np.ndarray,
                           save_path: str = None,
                           alpha: float = 0.6) -> plt.Figure:
        """
        Visualize segmentation prediction with overlay.
        
        Args:
            image: Original image
            mask: Ground truth mask
            prediction: Predicted mask
            save_path: Path to save visualization
            alpha: Transparency for overlay
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Ground truth mask
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        # Predicted mask
        axes[0, 2].imshow(prediction, cmap='gray')
        axes[0, 2].set_title('Prediction')
        axes[0, 2].axis('off')
        
        # Overlay: Original + Ground Truth
        overlay_gt = image.copy()
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0.5] = [0, 255, 0]  # Green for ground truth
        overlay_gt = cv2.addWeighted(overlay_gt, 1-alpha, mask_colored, alpha, 0)
        axes[1, 0].imshow(overlay_gt)
        axes[1, 0].set_title('Original + Ground Truth')
        axes[1, 0].axis('off')
        
        # Overlay: Original + Prediction
        overlay_pred = image.copy()
        pred_colored = np.zeros_like(image)
        pred_colored[prediction > 0.5] = [255, 0, 0]  # Red for prediction
        overlay_pred = cv2.addWeighted(overlay_pred, 1-alpha, pred_colored, alpha, 0)
        axes[1, 1].imshow(overlay_pred)
        axes[1, 1].set_title('Original + Prediction')
        axes[1, 1].axis('off')
        
        # Comparison
        comparison = image.copy()
        comparison[mask > 0.5] = [0, 255, 0]  # Green for ground truth
        comparison[prediction > 0.5] = [255, 0, 0]  # Red for prediction
        # Yellow for overlap
        overlap = (mask > 0.5) & (prediction > 0.5)
        comparison[overlap] = [255, 255, 0]
        axes[1, 2].imshow(comparison)
        axes[1, 2].set_title('Comparison (Green=GT, Red=Pred, Yellow=Overlap)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def create_medical_visualization(self, 
                                   image: np.ndarray,
                                   mask: np.ndarray,
                                   prediction: np.ndarray,
                                   save_path: str = None) -> plt.Figure:
        """
        Create medical-style visualization with annotations.
        
        Args:
            image: Medical image
            mask: Ground truth segmentation
            prediction: Predicted segmentation
            save_path: Path to save visualization
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Create medical-style colormap
        colors = ['black', 'red', 'yellow', 'white']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('medical', colors, N=n_bins)
        
        # Original image with medical colormap
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Medical Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Ground truth with medical colormap
        axes[1].imshow(mask, cmap=cmap, vmin=0, vmax=1)
        axes[1].set_title('Ground Truth Segmentation', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Prediction with medical colormap
        axes[2].imshow(prediction, cmap=cmap, vmin=0, vmax=1)
        axes[2].set_title('Predicted Segmentation', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(axes[2].images[0], ax=axes[2], fraction=0.046, pad=0.04)
        cbar.set_label('Segmentation Confidence', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


class UncertaintyVisualizer:
    """
    Visualize model uncertainty and confidence.
    """
    
    def __init__(self, save_dir: str = "uncertainty"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def visualize_uncertainty(self, 
                            prediction: np.ndarray,
                            uncertainty: np.ndarray,
                            save_path: str = None) -> plt.Figure:
        """
        Visualize prediction uncertainty.
        
        Args:
            prediction: Model prediction
            uncertainty: Uncertainty map
            save_path: Path to save visualization
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Prediction
        axes[0].imshow(prediction, cmap='viridis')
        axes[0].set_title('Prediction')
        axes[0].axis('off')
        
        # Uncertainty
        im = axes[1].imshow(uncertainty, cmap='hot')
        axes[1].set_title('Uncertainty Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # Overlay
        overlay = prediction.copy()
        overlay_colored = plt.cm.viridis(overlay)
        uncertainty_colored = plt.cm.hot(uncertainty)
        
        # Blend based on uncertainty
        alpha = uncertainty / uncertainty.max()
        blended = (1 - alpha[:, :, np.newaxis]) * overlay_colored[:, :, :3] + \
                  alpha[:, :, np.newaxis] * uncertainty_colored[:, :, :3]
        
        axes[2].imshow(blended)
        axes[2].set_title('Prediction + Uncertainty')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


def create_comprehensive_report(history: Dict[str, List[float]],
                              metrics: Dict[str, float],
                              save_dir: str = "reports") -> str:
    """
    Create a comprehensive training report with all visualizations.
    
    Args:
        history: Training history
        metrics: Final metrics
        save_dir: Directory to save report
        
    Returns:
        Path to saved report
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    visualizer = TrainingVisualizer(save_dir)
    
    # Training curves
    visualizer.plot_training_curves(history, save_dir / "training_curves.png")
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>UNet Training Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
            .metric-card {{ background-color: #e8f4f8; padding: 15px; border-radius: 8px; text-align: center; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ font-size: 14px; color: #7f8c8d; }}
            img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>UNet Training Report</h1>
            <p>Comprehensive analysis of model training and performance</p>
        </div>
        
        <h2>Final Metrics</h2>
        <div class="metrics">
    """
    
    for metric, value in metrics.items():
        html_content += f"""
            <div class="metric-card">
                <div class="metric-value">{value:.4f}</div>
                <div class="metric-label">{metric.upper()}</div>
            </div>
        """
    
    html_content += f"""
        </div>
        
        <h2>Training Curves</h2>
        <img src="training_curves.png" alt="Training Curves">
        
        <h2>Training Summary</h2>
        <ul>
            <li>Total epochs: {len(history.get('train_loss', []))}</li>
            <li>Best validation loss: {min(history.get('val_loss', [float('inf')])):.4f}</li>
            <li>Best validation dice: {max(history.get('val_dice', [0])):.4f}</li>
        </ul>
    </body>
    </html>
    """
    
    report_path = save_dir / "training_report.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
        
    return str(report_path) 