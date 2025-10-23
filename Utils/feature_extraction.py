import torch
import numpy as np
from models.superpoint import SuperPoint
from utils.image_processing import load_image, resize_image

def extract_superpoint_features(image_path, device='auto'):
    """Extract SuperPoint features from an image"""
    try:
        # Auto-detect device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load and preprocess image
        img = load_image(image_path)
        img = resize_image(img)
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img)[None, None].to(device)
        
        # Load model
        model = SuperPoint({}).to(device)
        model.eval()
        
        # Extract features
        with torch.no_grad():
            result = model({'image': img_tensor})
        
        # Convert to numpy arrays
        keypoints = result['keypoints'][0].cpu().numpy()
        descriptors = result['descriptors'][0].cpu().numpy()
        scores = result['scores'][0].cpu().numpy()
        
        return keypoints, descriptors, scores
        
    except Exception as e:
        raise RuntimeError(f"Failed to extract SuperPoint features: {str(e)}") 