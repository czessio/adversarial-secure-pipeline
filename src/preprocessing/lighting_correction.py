# Lighting correction algorithms including Retinex and histogram equalisation

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Union, Tuple, Optional
from torchvision import transforms


class RetinexCorrection:
    """Single-scale Retinex algorithm for lighting correction."""
    
    def __init__(self, sigma: float = 50.0):
        """
        Initialise Retinex correction.
        
        Args:
            sigma: Standard deviation for Gaussian filter
        """
        self.sigma = sigma
    
    def __call__(self, image: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Apply Retinex correction to image.
        
        Args:
            image: Input image (CHW format if tensor, HWC if numpy)
        
        Returns:
            Corrected image in same format as input
        """
        is_tensor = isinstance(image, torch.Tensor)
        
        if is_tensor:
            # Convert to numpy for processing
            device = image.device
            image_np = image.cpu().numpy()
            if image_np.ndim == 3:
                image_np = np.transpose(image_np, (1, 2, 0))
        else:
            # Handle PIL Image
            if hasattr(image, 'mode'):  # PIL Image
                image_np = np.array(image)
                is_tensor = False
            else:
                image_np = image.copy()
        
        # Ensure image is in float format
        if image_np.dtype != np.float32:
            image_np = image_np.astype(np.float32)
            if image_np.max() > 1:
                image_np /= 255.0
        
        # Apply Retinex to each channel
        corrected = np.zeros_like(image_np)
        for i in range(image_np.shape[2]):
            channel = image_np[:, :, i]
            
            # Apply Gaussian blur to estimate illumination
            illumination = cv2.GaussianBlur(channel, (0, 0), self.sigma)
            
            # Compute reflectance (avoid log of zero)
            reflectance = np.log10(channel + 1e-6) - np.log10(illumination + 1e-6)
            
            # Normalise to [0, 1]
            reflectance = (reflectance - reflectance.min()) / (reflectance.max() - reflectance.min() + 1e-6)
            corrected[:, :, i] = reflectance
        
        if is_tensor:
            # Convert back to tensor
            corrected = np.transpose(corrected, (2, 0, 1))
            corrected = torch.from_numpy(corrected).to(device)
        
        return corrected


class AdaptiveHistogramEqualisation:
    """Contrast Limited Adaptive Histogram Equalisation (CLAHE)."""
    
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        """
        Initialise CLAHE.
        
        Args:
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalisation
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, image: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Apply CLAHE to image.
        
        Args:
            image: Input image (CHW format if tensor, HWC if numpy)
        
        Returns:
            Equalised image in same format as input
        """
        is_tensor = isinstance(image, torch.Tensor)
        
        if is_tensor:
            device = image.device
            image_np = image.cpu().numpy()
            if image_np.ndim == 3:
                image_np = np.transpose(image_np, (1, 2, 0))
        else:
            # Handle PIL Image
            if hasattr(image, 'mode'):  # PIL Image
                image_np = np.array(image)
                is_tensor = False
            else:
                image_np = image.copy()
        
        # Convert to uint8 for CLAHE
        if image_np.dtype == np.float32 or image_np.dtype == np.float64:
            image_np = (image_np * 255).astype(np.uint8)
        
        # Convert to LAB colour space
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Convert back to float
        corrected = corrected.astype(np.float32) / 255.0
        
        if is_tensor:
            corrected = np.transpose(corrected, (2, 0, 1))
            corrected = torch.from_numpy(corrected).to(device)
        
        return corrected


class LightingCorrectionTransform:
    """Combined lighting correction transform for data pipeline."""
    
    def __init__(self, config: dict):
        """
        Initialise lighting correction transform.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config['preprocessing']['lighting_correction']
        self.transforms = []
        
        if self.config['enable_retinex']:
            self.transforms.append(RetinexCorrection(sigma=self.config['retinex_sigma']))
        
        if self.config['enable_histogram_eq']:
            self.transforms.append(AdaptiveHistogramEqualisation(
                clip_limit=self.config['clip_limit'],
                tile_grid_size=tuple(self.config['tile_grid_size'])
            ))
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply all enabled lighting corrections."""
        for transform in self.transforms:
            image = transform(image)
        return image


def create_lighting_corrected_transform(config: dict, base_transform: Optional[transforms.Compose] = None) -> transforms.Compose:
    """
    Create transform pipeline with lighting correction.
    
    Args:
        config: Configuration dictionary
        base_transform: Base transforms to apply after lighting correction
    
    Returns:
        Complete transform pipeline
    """
    transform_list = []
    
    # First, add transforms that come before ToTensor (if any)
    pre_tensor_transforms = []
    post_tensor_transforms = []
    to_tensor_transform = None
    
    if base_transform is not None:
        if isinstance(base_transform, transforms.Compose):
            found_to_tensor = False
            for t in base_transform.transforms:
                if isinstance(t, transforms.ToTensor):
                    to_tensor_transform = t
                    found_to_tensor = True
                elif not found_to_tensor:
                    pre_tensor_transforms.append(t)
                else:
                    post_tensor_transforms.append(t)
        else:
            transform_list.append(base_transform)
    
    # Add pre-tensor transforms
    transform_list.extend(pre_tensor_transforms)
    
    # Add ToTensor if not already present
    if to_tensor_transform is None:
        transform_list.append(transforms.ToTensor())
    else:
        transform_list.append(to_tensor_transform)
    
    # Add lighting correction after ToTensor
    if (config['preprocessing']['lighting_correction']['enable_retinex'] or 
        config['preprocessing']['lighting_correction']['enable_histogram_eq']):
        transform_list.append(LightingCorrectionTransform(config))
    
    # Add post-tensor transforms
    transform_list.extend(post_tensor_transforms)
    
    return transforms.Compose(transform_list)


def visualise_lighting_correction(image: np.ndarray, config: dict) -> dict:
    """
    Apply and visualise different lighting correction methods.
    
    Args:
        image: Input image (HWC format)
        config: Configuration dictionary
    
    Returns:
        Dictionary containing original and corrected images
    """
    results = {'original': image}
    
    # Apply Retinex
    if config['preprocessing']['lighting_correction']['enable_retinex']:
        retinex = RetinexCorrection(sigma=config['preprocessing']['lighting_correction']['retinex_sigma'])
        results['retinex'] = retinex(image)
    
    # Apply CLAHE
    if config['preprocessing']['lighting_correction']['enable_histogram_eq']:
        clahe = AdaptiveHistogramEqualisation(
            clip_limit=config['preprocessing']['lighting_correction']['clip_limit'],
            tile_grid_size=tuple(config['preprocessing']['lighting_correction']['tile_grid_size'])
        )
        results['clahe'] = clahe(image)
    
    # Apply both
    if len(results) > 2:
        combined = image
        for key in ['retinex', 'clahe']:
            if key in results:
                if key == 'retinex':
                    combined = retinex(combined)
                else:
                    combined = clahe(combined)
        results['combined'] = combined
    
    return results