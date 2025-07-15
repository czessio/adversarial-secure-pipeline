# Utility functions for adversarial attacks

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def normalise_perturbation(
    perturbation: torch.Tensor,
    norm_type: str = 'linf',
    epsilon: float = 0.03
) -> torch.Tensor:
    """
    Normalise perturbation based on specified norm.
    
    Args:
        perturbation: Perturbation tensor
        norm_type: Type of norm ('linf', 'l2', 'l1')
        epsilon: Maximum perturbation magnitude
    
    Returns:
        Normalised perturbation
    """
    if norm_type == 'linf':
        # L-infinity norm: clip to [-epsilon, epsilon]
        return torch.clamp(perturbation, -epsilon, epsilon)
    
    elif norm_type == 'l2':
        # L2 norm: scale to have L2 norm <= epsilon
        batch_size = perturbation.size(0)
        perturbation_flat = perturbation.view(batch_size, -1)
        l2_norm = torch.norm(perturbation_flat, p=2, dim=1, keepdim=True)
        scale = torch.min(torch.ones_like(l2_norm), epsilon / (l2_norm + 1e-10))
        return perturbation * scale.view(batch_size, 1, 1, 1)
    
    elif norm_type == 'l1':
        # L1 norm: scale to have L1 norm <= epsilon
        batch_size = perturbation.size(0)
        perturbation_flat = perturbation.view(batch_size, -1)
        l1_norm = torch.norm(perturbation_flat, p=1, dim=1, keepdim=True)
        scale = torch.min(torch.ones_like(l1_norm), epsilon / (l1_norm + 1e-10))
        return perturbation * scale.view(batch_size, 1, 1, 1)
    
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def calculate_perturbation_metrics(
    original: torch.Tensor,
    adversarial: torch.Tensor
) -> Dict[str, float]:
    """
    Calculate metrics for adversarial perturbations.
    
    Args:
        original: Original images
        adversarial: Adversarial images
    
    Returns:
        Dictionary of perturbation metrics
    """
    perturbation = adversarial - original
    batch_size = original.size(0)
    
    # Flatten for easier computation
    perturbation_flat = perturbation.view(batch_size, -1)
    
    # Calculate various norms
    linf_norm = torch.max(torch.abs(perturbation_flat), dim=1)[0].mean().item()
    l2_norm = torch.norm(perturbation_flat, p=2, dim=1).mean().item()
    l1_norm = torch.norm(perturbation_flat, p=1, dim=1).mean().item()
    l0_norm = (perturbation_flat != 0).float().sum(dim=1).mean().item()
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    mse = torch.mean((original - adversarial) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse)).item() if mse > 0 else float('inf')
    
    return {
        'linf_norm': linf_norm,
        'l2_norm': l2_norm,
        'l1_norm': l1_norm,
        'l0_norm': l0_norm,
        'psnr': psnr,
        'mean_perturbation': perturbation.mean().item(),
        'std_perturbation': perturbation.std().item()
    }


def visualise_adversarial_examples(
    original: torch.Tensor,
    adversarial: torch.Tensor,
    predictions_original: torch.Tensor,
    predictions_adversarial: torch.Tensor,
    class_names: List[str],
    num_examples: int = 8,
    save_path: Optional[str] = None
):
    """
    Visualise adversarial examples.
    
    Args:
        original: Original images
        adversarial: Adversarial images
        predictions_original: Predictions on original images
        predictions_adversarial: Predictions on adversarial images
        class_names: List of class names
        num_examples: Number of examples to visualise
        save_path: Path to save visualisation
    """
    num_examples = min(num_examples, original.size(0))
    
    fig = plt.figure(figsize=(15, 4 * num_examples))
    gs = GridSpec(num_examples, 4, width_ratios=[1, 1, 1, 0.5])
    
    for i in range(num_examples):
        # Original image
        ax1 = fig.add_subplot(gs[i, 0])
        img_original = original[i].cpu().numpy().transpose(1, 2, 0)
        img_original = (img_original - img_original.min()) / (img_original.max() - img_original.min())
        ax1.imshow(img_original)
        ax1.set_title(f'Original\n{class_names[predictions_original[i]]}')
        ax1.axis('off')
        
        # Adversarial image
        ax2 = fig.add_subplot(gs[i, 1])
        img_adversarial = adversarial[i].cpu().numpy().transpose(1, 2, 0)
        img_adversarial = (img_adversarial - img_adversarial.min()) / (img_adversarial.max() - img_adversarial.min())
        ax2.imshow(img_adversarial)
        ax2.set_title(f'Adversarial\n{class_names[predictions_adversarial[i]]}')
        ax2.axis('off')
        
        # Perturbation (amplified for visibility)
        ax3 = fig.add_subplot(gs[i, 2])
        perturbation = (adversarial[i] - original[i]).cpu().numpy().transpose(1, 2, 0)
        perturbation_vis = perturbation * 10 + 0.5  # Amplify and centre
        perturbation_vis = np.clip(perturbation_vis, 0, 1)
        ax3.imshow(perturbation_vis)
        ax3.set_title('Perturbation\n(10x amplified)')
        ax3.axis('off')
        
        # Perturbation heatmap
        ax4 = fig.add_subplot(gs[i, 3])
        perturbation_magnitude = np.linalg.norm(perturbation, axis=2)
        im = ax4.imshow(perturbation_magnitude, cmap='hot')
        ax4.set_title('Magnitude')
        ax4.axis('off')
        plt.colorbar(im, ax=ax4, fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_gradient_visualisation(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Generate gradient visualisation for target class.
    
    Args:
        model: Model to analyse
        image: Input image
        target_class: Target class for gradient
        device: Device to use
    
    Returns:
        Gradient tensor
    """
    model.eval()
    image = image.to(device).requires_grad_(True)
    
    # Forward pass
    output = model(image.unsqueeze(0))
    
    # Create one-hot target
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    
    # Backward pass
    output.backward(gradient=one_hot)
    
    # Get gradient
    gradient = image.grad.detach()
    
    return gradient


def transfer_attack_evaluation(
    source_model: nn.Module,
    target_models: List[nn.Module],
    attack_fn: callable,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device = torch.device('cpu')
) -> Dict[str, float]:
    """
    Evaluate transferability of adversarial examples.
    
    Args:
        source_model: Model used to generate adversarial examples
        target_models: List of target models to evaluate
        attack_fn: Attack function to use
        test_loader: Test data loader
        device: Device to use
    
    Returns:
        Transfer success rates
    """
    source_model.eval()
    for model in target_models:
        model.eval()
    
    transfer_rates = {f'model_{i}': 0 for i in range(len(target_models))}
    total_samples = 0
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        # Generate adversarial examples using source model
        adv_data = attack_fn(source_model, data, target)
        
        # Evaluate on target models
        for i, target_model in enumerate(target_models):
            with torch.no_grad():
                outputs = target_model(adv_data)
                predictions = outputs.argmax(dim=1)
                # Count successful transfers (misclassifications)
                transfer_rates[f'model_{i}'] += (~predictions.eq(target)).sum().item()
        
        total_samples += target.size(0)
    
    # Calculate transfer rates
    for key in transfer_rates:
        transfer_rates[key] = (transfer_rates[key] / total_samples) * 100
    
    return transfer_rates


class AdversarialDataset(torch.utils.data.Dataset):
    """Dataset wrapper for pre-generated adversarial examples."""
    
    def __init__(
        self,
        clean_data: torch.Tensor,
        adversarial_data: torch.Tensor,
        labels: torch.Tensor,
        transform: Optional[callable] = None
    ):
        """
        Initialise adversarial dataset.
        
        Args:
            clean_data: Clean images
            adversarial_data: Adversarial images
            labels: Labels
            transform: Optional transform to apply
        """
        self.clean_data = clean_data
        self.adversarial_data = adversarial_data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        clean_img = self.clean_data[idx]
        adv_img = self.adversarial_data[idx]
        label = self.labels[idx]
        
        if self.transform:
            clean_img = self.transform(clean_img)
            adv_img = self.transform(adv_img)
        
        return {
            'clean': clean_img,
            'adversarial': adv_img,
            'label': label
        }