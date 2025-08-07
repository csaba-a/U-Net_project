#!/usr/bin/env python3
"""
Test script to verify that all components of the UNet training framework are properly installed.
"""

import sys
import torch
import numpy as np

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from models.unet import UNet
        print("✓ UNet model imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import UNet: {e}")
        return False
    
    try:
        from utils.dataset import SegmentationDataset, get_training_augmentation
        print("✓ Dataset utilities imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import dataset utilities: {e}")
        return False
    
    try:
        from utils.losses import get_loss_function, DiceLoss, DiceBCELoss
        print("✓ Loss functions imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import loss functions: {e}")
        return False
    
    try:
        from utils.metrics import SegmentationMetrics, dice_coefficient
        print("✓ Metrics imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import metrics: {e}")
        return False
    
    return True


def test_model_creation():
    """Test that the UNet model can be created"""
    print("\nTesting model creation...")
    
    try:
        from models.unet import UNet
        
        # Test different configurations
        model1 = UNet(n_channels=3, n_classes=1, bilinear=False)
        print("✓ UNet with 3 input channels, 1 output class created")
        
        model2 = UNet(n_channels=1, n_classes=2, bilinear=True)
        print("✓ UNet with 1 input channel, 2 output classes, bilinear upsampling created")
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model1 = model1.to(device)
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        output = model1(dummy_input)
        
        print(f"✓ Forward pass successful. Input shape: {dummy_input.shape}, Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_loss_functions():
    """Test that loss functions work correctly"""
    print("\nTesting loss functions...")
    
    try:
        from utils.losses import get_loss_function, DiceLoss, DiceBCELoss, IoULoss, FocalLoss
        import torch.nn.functional as F
        
        # Create dummy data
        predictions = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        # Apply sigmoid to predictions to get probabilities for loss functions that need them
        predictions_sigmoid = torch.sigmoid(predictions)
        
        # Test different loss functions
        loss_functions = ['dice', 'dice_bce', 'iou', 'focal', 'bce']
        
        for loss_name in loss_functions:
            criterion = get_loss_function(loss_name)
            # Use sigmoided predictions for loss functions that expect probabilities
            if loss_name in ['bce', 'dice_bce', 'focal']:
                loss = criterion(predictions_sigmoid, targets)
            else:
                loss = criterion(predictions_sigmoid, targets)
            print(f"✓ {loss_name} loss computed: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        return False


def test_metrics():
    """Test that metrics work correctly"""
    print("\nTesting metrics...")
    
    try:
        from utils.metrics import SegmentationMetrics, dice_coefficient, iou_score
        
        # Create dummy data
        predictions = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        # Test individual metrics
        dice = dice_coefficient(targets.numpy(), torch.sigmoid(predictions).numpy())
        iou = iou_score(targets.numpy(), torch.sigmoid(predictions).numpy())
        
        print(f"✓ Dice coefficient: {dice:.4f}")
        print(f"✓ IoU score: {iou:.4f}")
        
        # Test metrics class
        metrics = SegmentationMetrics()
        metrics.update(targets, predictions)
        results = metrics.get_metrics()
        
        print(f"✓ Metrics class - Dice: {results['dice']:.4f}, IoU: {results['iou']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        return False


def test_data_augmentation():
    """Test that data augmentation works"""
    print("\nTesting data augmentation...")
    
    try:
        from utils.dataset import get_training_augmentation, get_validation_augmentation
        
        # Create dummy image and mask
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        
        # Test training augmentation
        train_transform = get_training_augmentation((256, 256))
        augmented = train_transform(image=image, mask=mask)
        
        print(f"✓ Training augmentation successful. Output shape: {augmented['image'].shape}")
        
        # Test validation augmentation
        val_transform = get_validation_augmentation((256, 256))
        augmented = val_transform(image=image, mask=mask)
        
        print(f"✓ Validation augmentation successful. Output shape: {augmented['image'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data augmentation test failed: {e}")
        return False


def test_pytorch_installation():
    """Test PyTorch installation and CUDA availability"""
    print("\nTesting PyTorch installation...")
    
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
        print(f"✓ Current GPU: {torch.cuda.get_device_name(0)}")
    
    return True


def main():
    """Run all tests"""
    print("UNet Training Framework - Installation Test")
    print("=" * 50)
    
    tests = [
        test_pytorch_installation,
        test_imports,
        test_model_creation,
        test_loss_functions,
        test_metrics,
        test_data_augmentation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The UNet training framework is ready to use.")
        print("\nNext steps:")
        print("1. Organize your data in the required directory structure")
        print("2. Run 'python example.py' to see a demonstration")
        print("3. Use 'python train.py' to train on your own data")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are properly installed.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 