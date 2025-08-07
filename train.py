import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.unet import UNet
from utils.dataset import SegmentationDataset, get_training_augmentation, get_validation_augmentation
from utils.losses import get_loss_function
from utils.metrics import SegmentationMetrics


def train_epoch(model, dataloader, criterion, optimizer, device, metrics):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    metrics.reset()
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        metrics.update(masks, outputs)
        
        # Update progress bar
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    epoch_metrics = metrics.get_metrics()
    
    return avg_loss, epoch_metrics


def validate_epoch(model, dataloader, criterion, device, metrics):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    metrics.reset()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Update metrics
            total_loss += loss.item()
            metrics.update(masks, outputs)
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    epoch_metrics = metrics.get_metrics()
    
    return avg_loss, epoch_metrics


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def plot_sample_predictions(model, dataloader, device, save_path, num_samples=4):
    """Plot sample predictions"""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            outputs = model(images)
            
            # Convert to numpy for plotting
            img = images[0].cpu().permute(1, 2, 0).numpy()
            mask = masks[0].cpu().squeeze().numpy()
            pred = torch.sigmoid(outputs[0]).cpu().squeeze().numpy()
            
            # Denormalize image
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
            
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    model = UNet(
        n_channels=args.input_channels,
        n_classes=args.num_classes,
        bilinear=args.bilinear
    ).to(device)
    
    # Initialize loss function
    criterion = get_loss_function(args.loss_function)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Initialize scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize metrics
    metrics = SegmentationMetrics(threshold=args.threshold)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(args.log_dir)
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume)
        print(f"Resuming from epoch {start_epoch}")
    
    # Create datasets
    train_transform = get_training_augmentation((args.image_size, args.image_size))
    val_transform = get_validation_augmentation((args.image_size, args.image_size))
    
    train_dataset = SegmentationDataset(
        args.train_images, args.train_masks, 
        transform=train_transform, 
        image_size=(args.image_size, args.image_size)
    )
    
    val_dataset = SegmentationDataset(
        args.val_images, args.val_masks, 
        transform=val_transform, 
        image_size=(args.image_size, args.image_size)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, metrics
        )
        
        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, metrics
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Dice/Train', train_metrics['dice'], epoch)
        writer.add_scalar('Dice/Val', val_metrics['dice'], epoch)
        writer.add_scalar('IoU/Train', train_metrics['iou'], epoch)
        writer.add_scalar('IoU/Val', val_metrics['iou'], epoch)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_metrics['dice']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_metrics['dice']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(args.checkpoint_dir, 'best_model.pth')
            )
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Save checkpoint every few epochs
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
        
        # Plot sample predictions
        if (epoch + 1) % args.plot_freq == 0:
            plot_sample_predictions(
                model, val_loader, device,
                os.path.join(args.output_dir, f'predictions_epoch_{epoch+1}.png')
            )
    
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UNet Training')
    
    # Data arguments
    parser.add_argument('--train-images', type=str, required=True, help='Path to training images')
    parser.add_argument('--train-masks', type=str, required=True, help='Path to training masks')
    parser.add_argument('--val-images', type=str, required=True, help='Path to validation images')
    parser.add_argument('--val-masks', type=str, required=True, help='Path to validation masks')
    
    # Model arguments
    parser.add_argument('--input-channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--num-classes', type=int, default=1, help='Number of output classes')
    parser.add_argument('--bilinear', action='store_true', help='Use bilinear upsampling')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--loss-function', type=str, default='dice_bce', 
                       choices=['dice', 'dice_bce', 'iou', 'focal', 'bce'], help='Loss function')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    
    # Data arguments
    parser.add_argument('--image-size', type=int, default=512, help='Image size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    
    # Output arguments
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--save-freq', type=int, default=10, help='Save frequency')
    parser.add_argument('--plot-freq', type=int, default=20, help='Plot frequency')
    
    # Resume training
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    main(args) 