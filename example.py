#!/usr/bin/env python3
"""
Example script demonstrating how to use the UNet training framework.
This script shows how to train a UNet model on a sample dataset.
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models.unet import UNet
from utils.dataset import SegmentationDataset, get_training_augmentation
from utils.losses import get_loss_function
from utils.metrics import SegmentationMetrics


def create_sample_data(num_samples=100, image_size=256):
    """Create sample training data for demonstration"""
    print("Creating sample data...")
    
    # Create directories
    os.makedirs("data/train/images", exist_ok=True)
    os.makedirs("data/train/masks", exist_ok=True)
    os.makedirs("data/val/images", exist_ok=True)
    os.makedirs("data/val/masks", exist_ok=True)
    
    # Split into train/val
    train_samples = int(num_samples * 0.8)
    val_samples = num_samples - train_samples
    
    for i in range(num_samples):
        # Create random image
        image = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        
        # Create random mask (simple geometric shapes)
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        
        # Add some random circles
        for _ in range(np.random.randint(1, 5)):
            center_x = np.random.randint(50, image_size - 50)
            center_y = np.random.randint(50, image_size - 50)
            radius = np.random.randint(20, 60)
            
            y, x = np.ogrid[:image_size, :image_size]
            mask_circle = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            mask[mask_circle] = 255
        
        # Save image and mask
        if i < train_samples:
            img_path = f"data/train/images/sample_{i:03d}.png"
            mask_path = f"data/train/masks/sample_{i:03d}.png"
        else:
            img_path = f"data/val/images/sample_{i:03d}.png"
            mask_path = f"data/val/masks/sample_{i:03d}.png"
        
        Image.fromarray(image).save(img_path)
        Image.fromarray(mask).save(mask_path)
    
    print(f"Created {train_samples} training samples and {val_samples} validation samples")


def train_sample_model():
    """Train a sample UNet model"""
    print("Training sample UNet model...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
    
    # Create datasets
    train_transform = get_training_augmentation((256, 256))
    train_dataset = SegmentationDataset(
        "data/train/images", "data/train/masks", 
        transform=train_transform, image_size=(256, 256)
    )
    
    val_dataset = SegmentationDataset(
        "data/val/images", "data/val/masks", 
        transform=train_transform, image_size=(256, 256)
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=2
    )
    
    # Initialize training components
    criterion = get_loss_function('dice_bce')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    metrics = SegmentationMetrics()
    
    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0
        metrics.reset()
        
        for batch in train_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            metrics.update(masks, outputs)
        
        train_loss /= len(train_loader)
        train_metrics = metrics.get_metrics()
        
        # Validation
        model.eval()
        val_loss = 0
        metrics.reset()
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                metrics.update(masks, outputs)
        
        val_loss /= len(val_loader)
        val_metrics = metrics.get_metrics()
        
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_metrics['dice']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_metrics['dice']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, 'checkpoints/best_model.pth')
            print("Saved best model!")
    
    print("Training completed!")


def test_inference():
    """Test inference on a sample image"""
    print("Testing inference...")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
    
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load a sample image
    sample_image_path = "data/val/images/sample_080.png"
    if os.path.exists(sample_image_path):
        # Load and preprocess image
        image = Image.open(sample_image_path).convert('RGB')
        image = image.resize((256, 256))
        image_array = np.array(image)
        
        # Apply transformations
        transform = get_training_augmentation((256, 256))
        augmented = transform(image=image_array)
        image_tensor = augmented['image'].unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            prediction = torch.sigmoid(output)
            prediction = (prediction > 0.5).float()
        
        # Visualize results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image_array)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Load corresponding mask
        mask_path = sample_image_path.replace('images', 'masks')
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize((256, 256))
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
        
        axes[2].imshow(prediction.cpu().squeeze().numpy(), cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_prediction.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Sample prediction saved as 'sample_prediction.png'")
    else:
        print("Sample image not found. Skipping inference test.")


def main():
    """Main function to run the example"""
    print("UNet Training Example")
    print("=" * 50)
    
    # Create sample data
    create_sample_data(num_samples=100, image_size=256)
    
    # Create output directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Train model
    train_sample_model()
    
    # Test inference
    test_inference()
    
    print("\nExample completed successfully!")
    print("You can now use the trained model for your own data.")


if __name__ == "__main__":
    main() 