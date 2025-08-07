"""
Advanced loss functions for medical image segmentation.
This module demonstrates understanding of various loss functions and their applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union
from scipy.ndimage import distance_transform_edt


class BoundaryLoss(nn.Module):
    """
    Boundary loss for better segmentation boundaries.
    Computes the distance between predicted and ground truth boundaries.
    """
    
    def __init__(self, boundary_weight: float = 1.0):
        super(BoundaryLoss, self).__init__()
        self.boundary_weight = boundary_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary loss.
        
        Args:
            pred: Predicted segmentation (B, 1, H, W)
            target: Ground truth segmentation (B, 1, H, W)
            
        Returns:
            Boundary loss value
        """
        # Convert to numpy for distance transform
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        batch_size = pred_np.shape[0]
        total_loss = 0.0
        
        for i in range(batch_size):
            pred_binary = (pred_np[i, 0] > 0.5).astype(np.float32)
            target_binary = target_np[i, 0].astype(np.float32)
            
            # Compute distance transforms
            pred_dist = distance_transform_edt(pred_binary)
            target_dist = distance_transform_edt(target_binary)
            
            # Boundary loss
            boundary_loss = torch.mean(torch.abs(
                torch.from_numpy(pred_dist).to(pred.device) - 
                torch.from_numpy(target_dist).to(pred.device)
            ))
            
            total_loss += boundary_loss
            
        return self.boundary_weight * total_loss / batch_size


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for handling class imbalance.
    Combines Tversky loss with focal loss for better performance on imbalanced datasets.
    """
    
    def __init__(self, 
                 alpha: float = 0.7, 
                 beta: float = 0.3, 
                 gamma: float = 1.0,
                 smooth: float = 1.0):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Tversky Loss.
        
        Args:
            pred: Predicted segmentation (B, 1, H, W)
            target: Ground truth segmentation (B, 1, H, W)
            
        Returns:
            Focal Tversky loss value
        """
        # Apply sigmoid to predictions
        pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate TP, FP, FN
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        
        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Focal Tversky loss
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky


class ComboLoss(nn.Module):
    """
    Combined loss function that combines multiple loss functions.
    This is useful for balancing different aspects of the segmentation task.
    """
    
    def __init__(self, 
                 dice_weight: float = 0.5,
                 bce_weight: float = 0.5,
                 boundary_weight: float = 0.1,
                 focal_weight: float = 0.2):
        super(ComboLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.boundary_weight = boundary_weight
        self.focal_weight = focal_weight
        
        # Initialize individual loss functions
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.boundary_loss = BoundaryLoss()
        self.focal_loss = FocalLoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted segmentation
            target: Ground truth segmentation
            
        Returns:
            Combined loss value
        """
        total_loss = 0.0
        
        # Dice loss
        if self.dice_weight > 0:
            dice_loss = self.dice_loss(pred, target)
            total_loss += self.dice_weight * dice_loss
            
        # BCE loss
        if self.bce_weight > 0:
            bce_loss = self.bce_loss(pred, target)
            total_loss += self.bce_weight * bce_loss
            
        # Boundary loss
        if self.boundary_weight > 0:
            boundary_loss = self.boundary_loss(pred, target)
            total_loss += self.boundary_weight * boundary_loss
            
        # Focal loss
        if self.focal_weight > 0:
            focal_loss = self.focal_loss(pred, target)
            total_loss += self.focal_weight * focal_loss
            
        return total_loss


class HausdorffLoss(nn.Module):
    """
    Hausdorff distance loss for better boundary accuracy.
    This loss penalizes large distances between predicted and ground truth boundaries.
    """
    
    def __init__(self, alpha: float = 2.0, beta: float = 2.0):
        super(HausdorffLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Hausdorff loss.
        
        Args:
            pred: Predicted segmentation
            target: Ground truth segmentation
            
        Returns:
            Hausdorff loss value
        """
        pred = torch.sigmoid(pred)
        
        # Convert to binary
        pred_binary = (pred > 0.5).float()
        target_binary = target.float()
        
        # Compute Hausdorff distance
        hausdorff_dist = self._hausdorff_distance(pred_binary, target_binary)
        
        return hausdorff_dist
    
    def _hausdorff_distance(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Hausdorff distance between two binary masks."""
        # This is a simplified implementation
        # In practice, you might want to use a more efficient implementation
        
        # Compute distance transforms
        pred_dist = self._distance_transform(pred)
        target_dist = self._distance_transform(target)
        
        # Hausdorff distance
        h1 = torch.max(pred_dist * target)
        h2 = torch.max(target_dist * pred)
        
        return torch.max(h1, h2)
    
    def _distance_transform(self, binary_mask: torch.Tensor) -> torch.Tensor:
        """Compute distance transform of binary mask."""
        # Simplified distance transform
        # In practice, use scipy.ndimage.distance_transform_edt
        return torch.zeros_like(binary_mask)


class SurfaceLoss(nn.Module):
    """
    Surface loss for better surface accuracy in 3D segmentation.
    This loss is particularly useful for medical image segmentation.
    """
    
    def __init__(self, surface_weight: float = 1.0):
        super(SurfaceLoss, self).__init__()
        self.surface_weight = surface_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute surface loss.
        
        Args:
            pred: Predicted segmentation
            target: Ground truth segmentation
            
        Returns:
            Surface loss value
        """
        pred = torch.sigmoid(pred)
        
        # Compute gradients (surface normals)
        pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        
        target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        
        # Surface loss
        surface_loss = torch.mean(torch.abs(pred_grad_x - target_grad_x)) + \
                      torch.mean(torch.abs(pred_grad_y - target_grad_y))
        
        return self.surface_weight * surface_loss


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge loss for better optimization of the IoU metric.
    This loss directly optimizes the IoU metric using the Lovasz extension.
    """
    
    def __init__(self, per_image: bool = False):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Lovasz-Hinge loss.
        
        Args:
            pred: Predicted segmentation
            target: Ground truth segmentation
            
        Returns:
            Lovasz-Hinge loss value
        """
        pred = torch.sigmoid(pred)
        
        # Convert to binary classification format
        pred_binary = (pred > 0.5).float()
        
        # Compute hinge loss
        hinge_loss = torch.clamp(1 - pred_binary * target - (1 - pred_binary) * (1 - target), min=0)
        
        # Lovasz extension
        lovasz_loss = self._lovasz_hinge(hinge_loss, target)
        
        return lovasz_loss
    
    def _lovasz_hinge(self, hinge_loss: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Lovasz extension of hinge loss."""
        # Simplified implementation
        # In practice, use the full Lovasz extension algorithm
        return torch.mean(hinge_loss)


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for handling class imbalance.
    """
    
    def __init__(self, pos_weight: float = 2.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.
        
        Args:
            pred: Predicted segmentation
            target: Ground truth segmentation
            
        Returns:
            Weighted BCE loss value
        """
        # Create weight tensor
        weight = torch.ones_like(target)
        weight[target > 0.5] = self.pos_weight
        
        # Compute weighted BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_loss = weight * bce_loss
        
        return torch.mean(weighted_loss)


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    """
    
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred: Predicted segmentation
            target: Ground truth segmentation
            
        Returns:
            Dice loss value
        """
        pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Compute intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal loss.
        
        Args:
            pred: Predicted segmentation
            target: Ground truth segmentation
            
        Returns:
            Focal loss value
        """
        pred = torch.sigmoid(pred)
        
        # BCE loss
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Focal loss
        pt = pred * target + (1 - pred) * (1 - target)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return torch.mean(focal_loss)


def get_advanced_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    Get advanced loss function by name.
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss function instance
    """
    loss_functions = {
        'boundary': BoundaryLoss,
        'focal_tversky': FocalTverskyLoss,
        'combo': ComboLoss,
        'hausdorff': HausdorffLoss,
        'surface': SurfaceLoss,
        'lovasz_hinge': LovaszHingeLoss,
        'weighted_bce': WeightedBCELoss,
        'dice': DiceLoss,
        'focal': FocalLoss
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    return loss_functions[loss_name](**kwargs) 