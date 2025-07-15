# Robust classifier implementation with advanced defence mechanisms

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np

from .base_classifier import BaseClassifier


class RobustClassifier(BaseClassifier):
    """Classifier with built-in robustness mechanisms."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise robust classifier.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Additional robustness components
        self.use_input_randomisation = True
        self.use_feature_denoising = True
        
        # Input randomisation
        self.input_transforms = nn.ModuleList()
        
        # Feature denoising layers
        if self.use_feature_denoising:
            self._add_denoising_layers()
        
        # Lipschitz constraint weight
        self.lipschitz_constant = 1.0
    
    def _add_denoising_layers(self):
        """Add denoising layers to the model."""
        # Get feature dimension from backbone
        if hasattr(self.backbone, 'fc'):
            feature_dim = self.backbone.fc.in_features
        else:
            feature_dim = 512  # Default
        
        # Denoising autoencoder for features
        self.feature_denoiser = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Tanh()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        apply_randomisation: bool = True
    ) -> torch.Tensor:
        """
        Forward pass with optional robustness mechanisms.
        
        Args:
            x: Input tensor
            return_features: Whether to return intermediate features
            apply_randomisation: Whether to apply input randomisation
        
        Returns:
            Model output or (output, features) tuple
        """
        # Apply input randomisation during training
        if self.training and self.use_input_randomisation and apply_randomisation and len(self.input_transforms) > 0:
            # Randomly select and apply transformation
            if np.random.rand() > 0.5:
                transform_idx = np.random.randint(len(self.input_transforms))
                x = self.input_transforms[transform_idx](x)
        
        # Extract features
        features = self.get_features(x)
        
        # Apply feature denoising
        if self.use_feature_denoising:
            denoised_features = features + 0.1 * self.feature_denoiser(features)
            features = denoised_features
        
        # Classification
        output = self.classifier(features)
        
        if return_features:
            return output, features
        return output
    
    def spectral_normalisation_power_iteration(
        self,
        weight: torch.Tensor,
        u: Optional[torch.Tensor] = None,
        num_iterations: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply spectral normalisation using power iteration.
        
        Args:
            weight: Weight matrix
            u: Left singular vector
            num_iterations: Number of power iterations
        
        Returns:
            Spectral norm and updated u
        """
        if u is None:
            u = torch.randn(weight.size(0), 1, device=weight.device)
            u = F.normalize(u, p=2, dim=0)
        
        with torch.no_grad():
            for _ in range(num_iterations):
                v = F.normalize(torch.matmul(weight.t(), u), p=2, dim=0)
                u = F.normalize(torch.matmul(weight, v), p=2, dim=0)
        
        sigma = torch.matmul(torch.matmul(u.t(), weight), v)
        return sigma, u
    
    def apply_lipschitz_constraint(self):
        """Apply Lipschitz constraint to model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                weight = module.weight
                
                if isinstance(module, nn.Conv2d):
                    # Reshape conv weights for spectral norm calculation
                    weight = weight.view(weight.size(0), -1)
                
                # Calculate spectral norm
                sigma, _ = self.spectral_normalisation_power_iteration(weight)
                
                # Constrain weights if needed
                if sigma > self.lipschitz_constant:
                    module.weight.data = module.weight.data * (self.lipschitz_constant / sigma)


class CertifiedRobustClassifier(RobustClassifier):
    """Classifier with certified robustness guarantees."""
    
    def __init__(self, config: Dict[str, Any], sigma: float = 0.25):
        """
        Initialise certified robust classifier.
        
        Args:
            config: Configuration dictionary
            sigma: Noise level for randomised smoothing
        """
        super().__init__(config)
        self.sigma = sigma
        self.num_samples = 100  # Number of samples for prediction
        self.alpha = 0.001  # Confidence level
    
    def predict_with_certificate(
        self,
        x: torch.Tensor,
        n_samples: int = 1000,
        batch_size: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make prediction with robustness certificate.
        
        Args:
            x: Input tensor
            n_samples: Number of samples for smoothing
            batch_size: Batch size for sampling
        
        Returns:
            Predicted class and certified radius
        """
        self.eval()
        
        counts = torch.zeros(x.size(0), self.num_classes, device=x.device)
        
        for _ in range(0, n_samples, batch_size):
            this_batch_size = min(batch_size, n_samples)
            
            # Add Gaussian noise
            batch = x.unsqueeze(1).repeat(1, this_batch_size, 1, 1, 1)
            noise = torch.randn_like(batch) * self.sigma
            batch = batch + noise
            
            # Reshape for forward pass
            batch = batch.view(-1, *x.shape[1:])
            
            # Get predictions
            with torch.no_grad():
                outputs = self.forward(batch, apply_randomisation=False)
                predictions = outputs.argmax(dim=1)
            
            # Reshape and count
            predictions = predictions.view(x.size(0), this_batch_size)
            for i in range(x.size(0)):
                for j in range(this_batch_size):
                    counts[i, predictions[i, j]] += 1
        
        # Get top two classes
        counts_sorted, _ = counts.sort(dim=1, descending=True)
        top_class = counts.argmax(dim=1)
        
        # Calculate certified radius
        count_top = counts_sorted[:, 0]
        count_second = counts_sorted[:, 1]
        
        # Binomial confidence intervals
        from scipy.stats import binom
        radius = torch.zeros(x.size(0), device=x.device)
        
        for i in range(x.size(0)):
            nA = int(count_top[i].item())
            nB = int(count_second[i].item())
            
            if binom.cdf(nA, nA + nB, 0.5) <= self.alpha:
                pA = binom.ppf(1 - self.alpha, nA + nB, 0.5) / (nA + nB)
                radius[i] = self.sigma * torch.tensor(np.sqrt(2) * 
                    torch.erfinv(torch.tensor(2 * pA - 1))).item()
        
        return top_class, radius


class AdversariallyTrainedRobustClassifier(RobustClassifier):
    """Robust classifier specifically designed for adversarial training."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise adversarially trained robust classifier.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Additional components for adversarial robustness
        self.trades_beta = 6.0  # TRADES parameter
        
        # Gradient penalty weight
        self.gradient_penalty_weight = 0.1
        
        # Feature scatter regularisation
        self.feature_scatter_weight = 0.01
    
    def trades_loss(
        self,
        logits_clean: torch.Tensor,
        logits_adv: torch.Tensor,
        labels: torch.Tensor,
        beta: float = 6.0
    ) -> torch.Tensor:
        """
        Calculate TRADES loss.
        
        Args:
            logits_clean: Clean logits
            logits_adv: Adversarial logits
            labels: True labels
            beta: Trade-off parameter
        
        Returns:
            TRADES loss
        """
        # Natural loss
        loss_natural = F.cross_entropy(logits_clean, labels)
        
        # Robust loss (KL divergence)
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        loss_robust = criterion_kl(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits_clean, dim=1)
        )
        
        return loss_natural + beta * loss_robust
    
    def feature_scatter_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate feature scatter loss for better class separation.
        
        Args:
            features: Feature representations
            labels: True labels
        
        Returns:
            Feature scatter loss
        """
        batch_size = features.size(0)
        
        # Normalise features
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Calculate pairwise distances
        dist_matrix = torch.cdist(features_norm, features_norm, p=2)
        
        # Create same-class mask
        labels_expand = labels.unsqueeze(1).expand(batch_size, batch_size)
        same_class_mask = (labels_expand == labels_expand.t()).float()
        
        # Intra-class scatter (minimise)
        intra_class_dist = (dist_matrix * same_class_mask).sum() / (same_class_mask.sum() + 1e-6)
        
        # Inter-class scatter (maximise)
        diff_class_mask = 1 - same_class_mask
        inter_class_dist = (dist_matrix * diff_class_mask).sum() / (diff_class_mask.sum() + 1e-6)
        
        # Feature scatter loss
        return intra_class_dist - 0.1 * inter_class_dist
    
    def gradient_penalty(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate gradient penalty for input gradient regularisation.
        
        Args:
            x: Input tensor
            y: Output tensor
        
        Returns:
            Gradient penalty
        """
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty


def create_robust_model_architecture(
    config: Dict[str, Any],
    model_type: str = 'standard'
) -> nn.Module:
    """
    Create robust model architecture.
    
    Args:
        config: Configuration dictionary
        model_type: Type of robust model ('standard', 'certified', 'adversarial')
    
    Returns:
        Robust model instance
    """
    if model_type == 'standard':
        return RobustClassifier(config)
    elif model_type == 'certified':
        return CertifiedRobustClassifier(config)
    elif model_type == 'adversarial':
        return AdversariallyTrainedRobustClassifier(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")