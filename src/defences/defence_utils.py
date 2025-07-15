# Utility functions for adversarial defences

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from scipy.ndimage import gaussian_filter


class InputTransformation:
    """Base class for input transformation defences."""
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class JPEGCompression(InputTransformation):
    """JPEG compression defence."""
    
    def __init__(self, quality: int = 75):
        """
        Initialise JPEG compression defence.
        
        Args:
            quality: JPEG quality (1-100)
        """
        self.quality = quality
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply JPEG compression."""
        import io
        from PIL import Image
        import torchvision.transforms as transforms
        
        # Convert to PIL Image
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        
        compressed = []
        for img in x:
            # Convert to PIL
            pil_img = to_pil(img.cpu())
            
            # Compress using JPEG
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=self.quality)
            buffer.seek(0)
            
            # Load compressed image
            compressed_img = Image.open(buffer)
            compressed.append(to_tensor(compressed_img).to(x.device))
        
        return torch.stack(compressed)


class BitDepthReduction(InputTransformation):
    """Bit depth reduction defence."""
    
    def __init__(self, bits: int = 5):
        """
        Initialise bit depth reduction.
        
        Args:
            bits: Number of bits to keep
        """
        self.bits = bits
        self.levels = 2 ** bits
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bit depth reduction."""
        # Quantize to fewer levels
        x_quantized = torch.round(x * (self.levels - 1)) / (self.levels - 1)
        return x_quantized


class GaussianSmoothing(InputTransformation):
    """Gaussian smoothing defence."""
    
    def __init__(self, sigma: float = 0.5):
        """
        Initialise Gaussian smoothing.
        
        Args:
            sigma: Standard deviation for Gaussian kernel
        """
        self.sigma = sigma
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smoothing."""
        smoothed = []
        for img in x:
            img_np = img.cpu().numpy()
            smoothed_np = np.zeros_like(img_np)
            
            # Apply Gaussian filter to each channel
            for c in range(img_np.shape[0]):
                smoothed_np[c] = gaussian_filter(img_np[c], sigma=self.sigma)
            
            smoothed.append(torch.from_numpy(smoothed_np).to(x.device))
        
        return torch.stack(smoothed)


class RandomResizePadding(InputTransformation):
    """Random resize and padding defence."""
    
    def __init__(self, resize_ratio_range: Tuple[float, float] = (0.8, 1.0)):
        """
        Initialise random resize padding.
        
        Args:
            resize_ratio_range: Range of resize ratios
        """
        self.resize_ratio_range = resize_ratio_range
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random resize and padding."""
        batch_size, channels, height, width = x.shape
        
        # Random resize ratio
        ratio = np.random.uniform(*self.resize_ratio_range)
        new_height = int(height * ratio)
        new_width = int(width * ratio)
        
        # Resize
        x_resized = F.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=False)
        
        # Random padding
        pad_top = np.random.randint(0, height - new_height + 1)
        pad_left = np.random.randint(0, width - new_width + 1)
        pad_bottom = height - new_height - pad_top
        pad_right = width - new_width - pad_left
        
        # Apply padding
        x_padded = F.pad(x_resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        
        return x_padded


class DefensiveDistillation:
    """Defensive distillation for robust model training."""
    
    def __init__(self, temperature: float = 20.0):
        """
        Initialise defensive distillation.
        
        Args:
            temperature: Temperature for softmax
        """
        self.temperature = temperature
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        alpha: float = 0.7
    ) -> torch.Tensor:
        """
        Calculate distillation loss.
        
        Args:
            student_logits: Student model outputs
            teacher_logits: Teacher model outputs
            labels: True labels
            alpha: Weight for distillation loss
        
        Returns:
            Combined loss
        """
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Distillation loss
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            soft_targets,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard target loss
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        return alpha * distillation_loss + (1 - alpha) * hard_loss


class GradientMasking:
    """Gradient masking defence (Note: can give false sense of security)."""
    
    def __init__(self, mask_probability: float = 0.5):
        """
        Initialise gradient masking.
        
        Args:
            mask_probability: Probability of masking gradients
        """
        self.mask_probability = mask_probability
    
    def apply_mask(self, model: nn.Module):
        """Apply gradient masking to model."""
        for param in model.parameters():
            if param.grad is not None:
                mask = torch.bernoulli(
                    torch.full_like(param.grad, 1 - self.mask_probability)
                )
                param.grad *= mask


class EnsembleDefence:
    """Ensemble defence using multiple models."""
    
    def __init__(self, models: List[nn.Module], aggregation: str = 'average'):
        """
        Initialise ensemble defence.
        
        Args:
            models: List of models
            aggregation: Aggregation method ('average', 'majority_vote')
        """
        self.models = models
        self.aggregation = aggregation
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make ensemble prediction."""
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                output = model(x)
                
                if self.aggregation == 'average':
                    predictions.append(F.softmax(output, dim=1))
                else:  # majority_vote
                    predictions.append(output.argmax(dim=1))
        
        if self.aggregation == 'average':
            # Average probabilities
            avg_probs = torch.stack(predictions).mean(dim=0)
            return avg_probs.argmax(dim=1)
        else:
            # Majority voting
            votes = torch.stack(predictions)
            return torch.mode(votes, dim=0)[0]


class AdversarialDetector:
    """Detect adversarial examples using various methods."""
    
    def __init__(self, model: nn.Module, method: str = 'feature_statistics'):
        """
        Initialise adversarial detector.
        
        Args:
            model: Model to use for detection
            method: Detection method
        """
        self.model = model
        self.method = method
        self.clean_statistics = None
    
    def fit(self, clean_loader: torch.utils.data.DataLoader, device: torch.device):
        """Fit detector on clean data."""
        if self.method == 'feature_statistics':
            features = []
            
            self.model.eval()
            with torch.no_grad():
                for data, _ in clean_loader:
                    data = data.to(device)
                    # Extract features (assuming model has get_features method)
                    feat = self.model.get_features(data)
                    features.append(feat.cpu())
            
            features = torch.cat(features)
            self.clean_statistics = {
                'mean': features.mean(dim=0),
                'std': features.std(dim=0),
                'min': features.min(dim=0)[0],
                'max': features.max(dim=0)[0]
            }
    
    def detect(self, x: torch.Tensor, threshold: float = 3.0) -> torch.Tensor:
        """
        Detect adversarial examples.
        
        Args:
            x: Input batch
            threshold: Detection threshold
        
        Returns:
            Boolean tensor indicating adversarial examples
        """
        if self.method == 'feature_statistics' and self.clean_statistics is not None:
            self.model.eval()
            with torch.no_grad():
                features = self.model.get_features(x)
            
            # Calculate z-score
            z_score = torch.abs(
                (features.cpu() - self.clean_statistics['mean']) / 
                (self.clean_statistics['std'] + 1e-10)
            )
            
            # Flag as adversarial if any feature has high z-score
            is_adversarial = (z_score > threshold).any(dim=1)
            
            return is_adversarial
        else:
            raise NotImplementedError(f"Detection method {self.method} not implemented")


def evaluate_defence(
    model: nn.Module,
    defence: InputTransformation,
    test_loader: torch.utils.data.DataLoader,
    attack_fn: callable,
    device: torch.device = torch.device('cpu')
) -> Dict[str, float]:
    """
    Evaluate effectiveness of a defence.
    
    Args:
        model: Model to evaluate
        defence: Defence to apply
        test_loader: Test data loader
        attack_fn: Attack function
        device: Device to use
    
    Returns:
        Evaluation metrics
    """
    model.eval()
    
    clean_correct = 0
    defended_correct = 0
    total = 0
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        # Generate adversarial examples
        adv_data = attack_fn(model, data, target)
        
        # Apply defence
        defended_data = defence(adv_data)
        
        with torch.no_grad():
            # Clean accuracy
            clean_outputs = model(data)
            clean_pred = clean_outputs.argmax(dim=1)
            clean_correct += clean_pred.eq(target).sum().item()
            
            # Defended accuracy
            defended_outputs = model(defended_data)
            defended_pred = defended_outputs.argmax(dim=1)
            defended_correct += defended_pred.eq(target).sum().item()
            
            total += target.size(0)
    
    return {
        'clean_accuracy': 100. * clean_correct / total,
        'defended_accuracy': 100. * defended_correct / total,
        'defence_improvement': 100. * (defended_correct - clean_correct) / total
    }