import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def dice_coefficient(y_true, y_pred, smooth=1.0):
    """
    Calculate Dice coefficient
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def iou_score(y_true, y_pred, smooth=1.0):
    """
    Calculate IoU (Intersection over Union) score
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    Calculate multiple metrics for binary segmentation
    """
    # Convert to binary predictions
    y_pred_binary = (y_pred > threshold).astype(np.float32)
    y_true_binary = y_true.astype(np.float32)
    
    # Flatten arrays
    y_true_flat = y_true_binary.flatten()
    y_pred_flat = y_pred_binary.flatten()
    
    # Calculate metrics
    dice = dice_coefficient(y_true_binary, y_pred_binary)
    iou = iou_score(y_true_binary, y_pred_binary)
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    
    # Calculate precision, recall, and F1 score
    if np.sum(y_true_flat) > 0:  # Only if there are positive samples
        precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
        recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
        f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
    else:
        precision = recall = f1 = 0.0
    
    return {
        'dice': dice,
        'iou': iou,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


class SegmentationMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.dice_scores = []
        self.iou_scores = []
        self.accuracy_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.f1_scores = []
    
    def update(self, y_true, y_pred):
        """Update metrics with new batch"""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        
        # Apply sigmoid if needed
        if y_pred.max() > 1.0:
            y_pred = 1 / (1 + np.exp(-y_pred))
        
        metrics = calculate_metrics(y_true, y_pred, self.threshold)
        
        self.dice_scores.append(metrics['dice'])
        self.iou_scores.append(metrics['iou'])
        self.accuracy_scores.append(metrics['accuracy'])
        self.precision_scores.append(metrics['precision'])
        self.recall_scores.append(metrics['recall'])
        self.f1_scores.append(metrics['f1'])
    
    def get_metrics(self):
        """Get average metrics"""
        return {
            'dice': np.mean(self.dice_scores),
            'iou': np.mean(self.iou_scores),
            'accuracy': np.mean(self.accuracy_scores),
            'precision': np.mean(self.precision_scores),
            'recall': np.mean(self.recall_scores),
            'f1': np.mean(self.f1_scores)
        }
    
    def get_metrics_std(self):
        """Get standard deviation of metrics"""
        return {
            'dice_std': np.std(self.dice_scores),
            'iou_std': np.std(self.iou_scores),
            'accuracy_std': np.std(self.accuracy_scores),
            'precision_std': np.std(self.precision_scores),
            'recall_std': np.std(self.recall_scores),
            'f1_std': np.std(self.f1_scores)
        } 