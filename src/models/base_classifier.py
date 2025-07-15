# Base classifier model for image classification

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any


class BaseClassifier(nn.Module):
    """Base classifier with configurable architecture."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise base classifier.
        
        Args:
            config: Configuration dictionary
        """
        super(BaseClassifier, self).__init__()
        self.config = config
        self.num_classes = config['model']['num_classes']
        self.architecture = config['model']['architecture']
        self.dropout_rate = config['model']['dropout_rate']
        
        # Build model
        self._build_model()
    
    def _build_model(self):
        """Build the model architecture."""
        pretrained = self.config['model']['pretrained']
        
        if self.architecture == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif self.architecture == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif self.architecture == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif self.architecture == 'vgg16':
            self.backbone = models.vgg16(pretrained=pretrained)
            num_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        
        # Build classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, self.num_classes)
        )
        
        # Initialise weights
        self._initialise_weights()
    
    def _initialise_weights(self):
        """Initialise classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        return self.classifier(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the classifier head."""
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        return features
    
    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Get number of parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class EnsembleClassifier(nn.Module):
    """Ensemble of multiple classifiers for improved robustness."""
    
    def __init__(self, config: Dict[str, Any], num_models: int = 3):
        """
        Initialise ensemble classifier.
        
        Args:
            config: Configuration dictionary
            num_models: Number of models in ensemble
        """
        super(EnsembleClassifier, self).__init__()
        self.config = config
        self.num_models = num_models
        
        # Create multiple models with different initialisations
        self.models = nn.ModuleList([
            BaseClassifier(config) for _ in range(num_models)
        ])
    
    def forward(self, x: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            return_all: If True, return predictions from all models
        
        Returns:
            Averaged predictions or all predictions
        """
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        outputs = torch.stack(outputs)
        
        if return_all:
            return outputs
        
        # Return averaged predictions
        return outputs.mean(dim=0)
    
    def get_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction uncertainty based on ensemble variance.
        
        Args:
            x: Input tensor
        
        Returns:
            Uncertainty scores
        """
        outputs = self.forward(x, return_all=True)
        # Use softmax probabilities for uncertainty
        probs = torch.softmax(outputs, dim=-1)
        # Calculate variance across ensemble
        uncertainty = probs.var(dim=0).mean(dim=-1)
        return uncertainty


def create_model(config: Dict[str, Any], ensemble: bool = False) -> nn.Module:
    """
    Create model based on configuration.
    
    Args:
        config: Configuration dictionary
        ensemble: Whether to create an ensemble model
    
    Returns:
        Model instance
    """
    if ensemble:
        return EnsembleClassifier(config)
    return BaseClassifier(config)