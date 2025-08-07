import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        # Apply sigmoid to convert logits to probabilities
        inputs = torch.sigmoid(inputs)
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)  
        
        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight=0.5, smooth=1.0):
        super(DiceBCELoss, self).__init__()
        self.weight = weight
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        # Apply sigmoid to convert logits to probabilities for dice loss
        inputs_sigmoid = torch.sigmoid(inputs)
        
        # Flatten label and prediction tensors
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()                            
        dice_loss = 1 - (2.*intersection + self.smooth)/(inputs_flat.sum() + targets_flat.sum() + self.smooth)  
        
        # Use BCEWithLogitsLoss for better numerical stability
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        Dice_BCE = BCE * self.weight + dice_loss * (1 - self.weight)
        
        return Dice_BCE


class IoULoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        # Apply sigmoid to convert logits to probabilities
        inputs = torch.sigmoid(inputs)
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + self.smooth)/(union + self.smooth)
        return 1 - IoU


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, smooth=1.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        # Apply sigmoid to convert logits to probabilities
        inputs = torch.sigmoid(inputs)
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # BCE loss
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
        
        return focal_loss


def get_loss_function(loss_name='dice_bce'):
    """Get loss function by name"""
    if loss_name == 'dice':
        return DiceLoss()
    elif loss_name == 'dice_bce':
        return DiceBCELoss()
    elif loss_name == 'iou':
        return IoULoss()
    elif loss_name == 'focal':
        return FocalLoss()
    elif loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}") 