import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.unet import UNet
from utils.dataset import get_validation_augmentation


def load_model(checkpoint_path, device, n_channels=3, n_classes=1, bilinear=False):
    """Load trained model from checkpoint"""
    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def preprocess_image(image_path, image_size=512):
    """Preprocess image for inference"""
    # Load and resize image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((image_size, image_size))
    
    # Convert to numpy array
    image = np.array(image)
    
    # Apply transformations
    transform = get_validation_augmentation((image_size, image_size))
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0)  # Add batch dimension
    
    return image_tensor, image


def predict_single_image(model, image_tensor, device, threshold=0.5):
    """Make prediction for a single image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(output)
        
        # Convert to binary prediction
        prediction = (probabilities > threshold).float()
        
        return probabilities.cpu().numpy(), prediction.cpu().numpy()


def save_prediction(image, prediction, output_path, alpha=0.7):
    """Save prediction as overlay on original image"""
    # Convert prediction to RGB
    pred_rgb = np.stack([prediction.squeeze()] * 3, axis=-1)
    
    # Create overlay
    overlay = image * (1 - alpha) + (pred_rgb * 255 * alpha).astype(np.uint8)
    
    # Save result
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(prediction.squeeze(), cmap='gray')
    plt.title('Prediction')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay.astype(np.uint8))
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def predict_batch(model, image_dir, output_dir, device, image_size=512, threshold=0.5):
    """Predict on all images in a directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(image_dir) 
                   if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    print(f"Found {len(image_files)} images to process")
    
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_dir, image_file)
        
        try:
            # Preprocess image
            image_tensor, original_image = preprocess_image(image_path, image_size)
            
            # Make prediction
            probabilities, prediction = predict_single_image(
                model, image_tensor, device, threshold
            )
            
            # Save prediction
            output_path = os.path.join(output_dir, f"pred_{image_file}")
            save_prediction(original_image, prediction, output_path)
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(
        args.checkpoint_path, device, 
        n_channels=args.input_channels,
        n_classes=args.num_classes,
        bilinear=args.bilinear
    )
    print("Model loaded successfully!")
    
    if args.single_image:
        # Predict on single image
        print(f"Processing single image: {args.single_image}")
        
        # Preprocess image
        image_tensor, original_image = preprocess_image(args.single_image, args.image_size)
        
        # Make prediction
        probabilities, prediction = predict_single_image(
            model, image_tensor, device, args.threshold
        )
        
        # Save prediction
        output_path = args.output_path or "prediction.png"
        save_prediction(original_image, prediction, output_path)
        
        print(f"Prediction saved to: {output_path}")
        
    elif args.image_dir:
        # Predict on directory of images
        print(f"Processing images in directory: {args.image_dir}")
        predict_batch(
            model, args.image_dir, args.output_dir, device,
            args.image_size, args.threshold
        )
        print(f"Predictions saved to: {args.output_dir}")
    
    else:
        print("Please specify either --single-image or --image-dir")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UNet Inference')
    
    # Model arguments
    parser.add_argument('--checkpoint-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input-channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--num-classes', type=int, default=1, help='Number of output classes')
    parser.add_argument('--bilinear', action='store_true', help='Use bilinear upsampling')
    
    # Input arguments (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--single-image', type=str, help='Path to single image for prediction')
    group.add_argument('--image-dir', type=str, help='Directory containing images for prediction')
    
    # Output arguments
    parser.add_argument('--output-path', type=str, help='Output path for single image prediction')
    parser.add_argument('--output-dir', type=str, default='predictions', help='Output directory for batch predictions')
    
    # Processing arguments
    parser.add_argument('--image-size', type=int, default=512, help='Image size for processing')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    
    args = parser.parse_args()
    main(args) 