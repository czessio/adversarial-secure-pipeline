# Data augmentation utilities for improving model robustness

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, List, Optional, Union
import random


class MixupAugmentation:
    """Mixup data augmentation for improved robustness."""
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialise Mixup augmentation.
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply Mixup augmentation.
        
        Args:
            images: Batch of images
            labels: Batch of labels
        
        Returns:
            Mixed images, original labels, shuffled labels, lambda value
        """
        batch_size = images.size(0)
        
        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        # Random permutation for mixing
        index = torch.randperm(batch_size).to(images.device)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[index]
        
        # Return mixed images and both sets of labels
        return mixed_images, labels, labels[index], lam


class CutMixAugmentation:
    """CutMix data augmentation for improved robustness."""
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialise CutMix augmentation.
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def _rand_bbox(
        self,
        size: Tuple[int, int, int, int],
        lam: float
    ) -> Tuple[int, int, int, int]:
        """Generate random bounding box for CutMix."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix augmentation.
        
        Args:
            images: Batch of images
            labels: Batch of labels
        
        Returns:
            Mixed images, original labels, shuffled labels, lambda value
        """
        batch_size = images.size(0)
        
        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        # Random permutation for mixing
        index = torch.randperm(batch_size).to(images.device)
        
        # Generate random box
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        
        # Apply CutMix
        mixed_images = images.clone()
        mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        
        return mixed_images, labels, labels[index], lam


class AutoAugmentPolicy:
    """AutoAugment policy for CIFAR-10/100."""
    
    def __init__(self, dataset: str = 'CIFAR10'):
        """
        Initialise AutoAugment policy.
        
        Args:
            dataset: Dataset name ('CIFAR10' or 'CIFAR100')
        """
        self.dataset = dataset
        self.policies = self._get_policies()
    
    def _get_policies(self) -> List[List[Tuple[str, float, int]]]:
        """Get augmentation policies for dataset."""
        if self.dataset in ['CIFAR10', 'CIFAR100']:
            # CIFAR policies from AutoAugment paper
            return [
                [('Invert', 0.1, 7), ('Contrast', 0.2, 6)],
                [('Rotate', 0.7, 2), ('TranslateX', 0.3, 9)],
                [('Sharpness', 0.8, 1), ('Sharpness', 0.9, 3)],
                [('ShearY', 0.5, 8), ('TranslateY', 0.7, 9)],
                [('AutoContrast', 0.5, 8), ('Equalize', 0.9, 2)],
                [('ShearY', 0.2, 7), ('Posterize', 0.3, 7)],
                [('Colour', 0.4, 3), ('Brightness', 0.6, 7)],
                [('Sharpness', 0.3, 9), ('Brightness', 0.7, 9)],
                [('Equalize', 0.6, 5), ('Equalize', 0.5, 1)],
                [('Contrast', 0.6, 7), ('Sharpness', 0.6, 5)],
                [('Colour', 0.7, 7), ('TranslateX', 0.5, 8)],
                [('Equalize', 0.8, 8), ('Invert', 0.1, 3)],
                [('TranslateY', 0.4, 3), ('Sharpness', 0.2, 6)],
                [('Brightness', 0.9, 6), ('Colour', 0.2, 8)],
                [('Solarize', 0.5, 2), ('Invert', 0.0, 3)],
                [('Equalize', 0.2, 0), ('AutoContrast', 0.6, 0)],
                [('Equalize', 0.2, 8), ('Equalize', 0.6, 4)],
                [('Colour', 0.9, 9), ('Equalize', 0.6, 6)],
                [('AutoContrast', 0.8, 4), ('Solarize', 0.2, 8)],
                [('Brightness', 0.1, 3), ('Colour', 0.7, 0)],
                [('Solarize', 0.4, 5), ('AutoContrast', 0.9, 3)],
                [('TranslateY', 0.9, 9), ('TranslateY', 0.7, 9)],
                [('AutoContrast', 0.9, 2), ('Solarize', 0.8, 3)],
                [('Equalize', 0.8, 8), ('Invert', 0.1, 3)],
                [('TranslateY', 0.7, 9), ('AutoContrast', 0.9, 1)]
            ]
        else:
            # Default policies for other datasets
            return [
                [('Rotate', 0.5, 30), ('Colour', 0.5, 5)],
                [('Brightness', 0.5, 5), ('Contrast', 0.5, 5)],
                [('Sharpness', 0.5, 5), ('AutoContrast', 0.5, 5)]
            ]
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random policy to image."""
        policy = random.choice(self.policies)
        
        for (op_name, prob, magnitude) in policy:
            if random.random() < prob:
                image = self._apply_augmentation(image, op_name, magnitude)
        
        return image
    
    def _apply_augmentation(
        self,
        image: torch.Tensor,
        op_name: str,
        magnitude: int
    ) -> torch.Tensor:
        """Apply single augmentation operation."""
        # Convert magnitude to appropriate range
        magnitude = magnitude / 10.0
        
        # Some operations require uint8 tensors
        requires_uint8 = ['Equalize', 'Invert', 'AutoContrast', 'Solarize', 'Posterize']
        
        if op_name in requires_uint8:
            # Convert to uint8 for these operations
            was_float = image.dtype == torch.float32
            if was_float:
                image_uint8 = (image * 255).byte()
            else:
                image_uint8 = image
            
            if op_name == 'AutoContrast':
                result = transforms.functional.autocontrast(image_uint8)
            elif op_name == 'Equalize':
                result = transforms.functional.equalize(image_uint8)
            elif op_name == 'Invert':
                result = transforms.functional.invert(image_uint8)
            elif op_name == 'Solarize':
                threshold = int(255 * (1 - magnitude))
                result = transforms.functional.solarize(image_uint8, threshold)
            elif op_name == 'Posterize':
                bits = max(1, int(8 - magnitude * 4))  # Ensure at least 1 bit
                result = transforms.functional.posterize(image_uint8, bits)
            
            # Convert back to float if needed
            if was_float:
                return result.float() / 255.0
            return result
        
        # Operations that work with float tensors
        elif op_name == 'Rotate':
            angle = magnitude * 30  # Max 30 degrees
            return transforms.functional.rotate(image, angle)
        elif op_name == 'TranslateX':
            pixels = int(magnitude * image.shape[-1] * 0.3)
            return transforms.functional.affine(image, 0, (pixels, 0), 1, 0)
        elif op_name == 'TranslateY':
            pixels = int(magnitude * image.shape[-2] * 0.3)
            return transforms.functional.affine(image, 0, (0, pixels), 1, 0)
        elif op_name == 'Brightness':
            return transforms.functional.adjust_brightness(image, 1 + magnitude)
        elif op_name == 'Colour':
            return transforms.functional.adjust_saturation(image, 1 + magnitude)
        elif op_name == 'Contrast':
            return transforms.functional.adjust_contrast(image, 1 + magnitude)
        elif op_name == 'Sharpness':
            return transforms.functional.adjust_sharpness(image, 1 + magnitude)
        elif op_name == 'ShearX' or op_name == 'ShearY':
            shear = magnitude * 30  # Max 30 degrees
            if op_name == 'ShearX':
                return transforms.functional.affine(image, 0, (0, 0), 1, (shear, 0))
            else:
                return transforms.functional.affine(image, 0, (0, 0), 1, (0, shear))
        else:
            return image


class AugmentationWrapper(nn.Module):
    """Wrapper to apply augmentations during training."""
    
    def __init__(
        self,
        config: dict,
        use_mixup: bool = True,
        use_cutmix: bool = True,
        use_autoaugment: bool = True,
        mixup_alpha: float = 1.0,
        cutmix_alpha: float = 1.0
    ):
        """
        Initialise augmentation wrapper.
        
        Args:
            config: Configuration dictionary
            use_mixup: Whether to use Mixup
            use_cutmix: Whether to use CutMix
            use_autoaugment: Whether to use AutoAugment
            mixup_alpha: Mixup alpha parameter
            cutmix_alpha: CutMix alpha parameter
        """
        super().__init__()
        self.config = config
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.use_autoaugment = use_autoaugment
        
        self.mixup = MixupAugmentation(mixup_alpha) if use_mixup else None
        self.cutmix = CutMixAugmentation(cutmix_alpha) if use_cutmix else None
        self.autoaugment = AutoAugmentPolicy(config['data']['dataset_name']) if use_autoaugment else None
    
    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[float]]:
        """
        Apply augmentations.
        
        Args:
            images: Batch of images
            labels: Batch of labels
            training: Whether in training mode
        
        Returns:
            Augmented images, labels, mixed labels (optional), lambda (optional)
        """
        if not training:
            return images, labels, None, None
        
        # Apply AutoAugment first (per-image augmentation)
        if self.use_autoaugment and self.autoaugment is not None:
            augmented_images = []
            for img in images:
                augmented_images.append(self.autoaugment(img))
            images = torch.stack(augmented_images)
        
        # Apply Mixup or CutMix (batch augmentation)
        mixed_labels = None
        lam = None
        
        if (self.use_mixup or self.use_cutmix) and random.random() > 0.5:
            if self.use_cutmix and random.random() > 0.5:
                images, labels_a, labels_b, lam = self.cutmix(images, labels)
            else:
                images, labels_a, labels_b, lam = self.mixup(images, labels)
            
            # For mixed augmentations, we'll return both label sets
            mixed_labels = (labels_a, labels_b)
            labels = labels_a  # Primary labels
        
        return images, labels, mixed_labels, lam


def create_robust_augmentation_transform(config: dict) -> transforms.Compose:
    """
    Create augmentation transform for robust training.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Composed transform
    """
    transform_list = []
    
    # Add base augmentations
    aug_config = config['preprocessing']['augmentation']
    
    if aug_config['random_crop']:
        transform_list.append(transforms.RandomCrop(32, padding=4))
    
    if aug_config['random_horizontal_flip']:
        transform_list.append(transforms.RandomHorizontalFlip())
    
    # Add RandAugment
    transform_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
    
    # Add colour jitter
    if any(aug_config['colour_jitter'].values()):
        transform_list.append(transforms.ColorJitter(
            brightness=aug_config['colour_jitter']['brightness'],
            contrast=aug_config['colour_jitter']['contrast'],
            saturation=aug_config['colour_jitter']['saturation'],
            hue=aug_config['colour_jitter']['hue']
        ))
    
    # Add random erasing (cutout)
    transform_list.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)))
    
    return transforms.Compose(transform_list)