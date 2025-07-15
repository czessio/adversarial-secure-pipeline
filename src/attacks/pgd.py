# Projected Gradient Descent (PGD) adversarial attack implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import numpy as np


class PGDAttack:
    """Projected Gradient Descent attack."""
    
    def __init__(
        self,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        num_steps: int = 40,
        random_start: bool = True,
        targeted: bool = False,
        clip_min: float = 0.0,
        clip_max: float = 1.0
    ):
        """
        Initialise PGD attack.
        
        Args:
            epsilon: Maximum perturbation magnitude
            alpha: Step size for each iteration
            num_steps: Number of attack iterations
            random_start: Whether to start from random perturbation
            targeted: Whether to perform targeted attack
            clip_min: Minimum pixel value
            clip_max: Maximum pixel value
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start
        self.targeted = targeted
        self.clip_min = clip_min
        self.clip_max = clip_max
    
    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate adversarial examples using PGD.
        
        Args:
            model: Target model
            x: Input images
            y: True labels
            target: Target labels for targeted attack
        
        Returns:
            Adversarial examples
        """
        model.eval()
        
        # Initialise adversarial examples
        if self.random_start:
            # Random start within epsilon ball
            delta = torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x + delta, self.clip_min, self.clip_max)
        else:
            x_adv = x.clone()
        
        for step in range(self.num_steps):
            x_adv.requires_grad_(True)
            
            # Forward pass
            outputs = model(x_adv)
            
            # Calculate loss
            if self.targeted and target is not None:
                loss = -F.cross_entropy(outputs, target)
            else:
                loss = F.cross_entropy(outputs, y)
            
            # Calculate gradients
            loss.backward()
            grad = x_adv.grad.detach()
            
            # Update adversarial examples
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            
            # Project back to epsilon ball around original input
            delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = x + delta
            
            # Ensure valid pixel range
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        
        return x_adv.detach()
    
    def generate_batch(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device = torch.device('cpu')
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate adversarial examples for entire batch.
        
        Args:
            model: Target model
            data_loader: Data loader
            device: Device to use
        
        Returns:
            Tuple of (adversarial examples, true labels, predictions on adversarial examples)
        """
        model.eval()
        
        all_adv_examples = []
        all_labels = []
        all_adv_preds = []
        
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            # Generate adversarial examples
            adv_data = self.generate(model, data, target)
            
            # Get predictions on adversarial examples
            with torch.no_grad():
                adv_outputs = model(adv_data)
                adv_preds = adv_outputs.argmax(dim=1)
            
            all_adv_examples.append(adv_data.cpu())
            all_labels.append(target.cpu())
            all_adv_preds.append(adv_preds.cpu())
        
        return (
            torch.cat(all_adv_examples),
            torch.cat(all_labels),
            torch.cat(all_adv_preds)
        )


class MultiTargetedPGD(PGDAttack):
    """PGD attack that tries multiple target classes."""
    
    def __init__(
        self,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        num_steps: int = 40,
        num_classes: int = 10,
        num_targets: int = 9,
        random_start: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0
    ):
        """
        Initialise multi-targeted PGD attack.
        
        Args:
            epsilon: Maximum perturbation magnitude
            alpha: Step size for each iteration
            num_steps: Number of attack iterations
            num_classes: Total number of classes
            num_targets: Number of target classes to try
            random_start: Whether to start from random perturbation
            clip_min: Minimum pixel value
            clip_max: Maximum pixel value
        """
        super().__init__(epsilon, alpha, num_steps, random_start, True, clip_min, clip_max)
        self.num_classes = num_classes
        self.num_targets = min(num_targets, num_classes - 1)
    
    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate adversarial examples trying multiple targets."""
        model.eval()
        
        batch_size = x.size(0)
        best_adv = x.clone()
        best_loss = torch.zeros(batch_size).to(x.device)
        
        # Try different target classes
        for i in range(self.num_targets):
            # Select target classes different from true labels
            if target is None:
                target_labels = torch.zeros_like(y)
                for j in range(batch_size):
                    available = list(range(self.num_classes))
                    available.remove(y[j].item())
                    target_labels[j] = np.random.choice(available)
            else:
                target_labels = target
            
            # Run targeted PGD
            x_adv = super().generate(model, x, y, target_labels)
            
            # Evaluate attack success
            with torch.no_grad():
                outputs = model(x_adv)
                loss = F.cross_entropy(outputs, y, reduction='none')
                
                # Keep best adversarial examples (highest loss)
                mask = loss > best_loss
                best_adv[mask] = x_adv[mask]
                best_loss[mask] = loss[mask]
        
        return best_adv


def evaluate_pgd_robustness(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    epsilon_values: list,
    device: torch.device = torch.device('cpu'),
    num_steps: int = 40,
    alpha_ratio: float = 2.5
) -> dict:
    """
    Evaluate model robustness against PGD at different epsilon values.
    
    Args:
        model: Model to evaluate
        data_loader: Test data loader
        epsilon_values: List of epsilon values to test
        device: Device to use
        num_steps: Number of PGD steps
        alpha_ratio: Ratio of epsilon to alpha (alpha = epsilon / alpha_ratio)
    
    Returns:
        Dictionary with robustness metrics
    """
    results = {
        'epsilon': epsilon_values,
        'accuracy': [],
        'attack_success_rate': []
    }
    
    for eps in epsilon_values:
        alpha = eps / alpha_ratio
        pgd = PGDAttack(epsilon=eps, alpha=alpha, num_steps=num_steps)
        
        correct = 0
        successful_attacks = 0
        total = 0
        
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Generate adversarial examples
            adv_data = pgd.generate(model, data, target)
            
            # Evaluate on adversarial examples
            with torch.no_grad():
                outputs = model(adv_data)
                pred = outputs.argmax(dim=1)
                
                # Count correct predictions
                correct += pred.eq(target).sum().item()
                
                # Count successful attacks (misclassification)
                successful_attacks += (~pred.eq(target)).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        attack_success_rate = 100. * successful_attacks / total
        
        results['accuracy'].append(accuracy)
        results['attack_success_rate'].append(attack_success_rate)
        
        print(f"Epsilon: {eps:.3f} - Accuracy: {accuracy:.2f}% - Attack Success: {attack_success_rate:.2f}%")
    
    return results


def compare_attacks(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.03,
    device: torch.device = torch.device('cpu')
) -> dict:
    """
    Compare different attack methods on same inputs.
    
    Args:
        model: Target model
        x: Input images
        y: True labels
        epsilon: Maximum perturbation magnitude
        device: Device to use
    
    Returns:
        Dictionary with comparison results
    """
    from .fgsm import FGSMAttack, IterativeFGSM
    
    x, y = x.to(device), y.to(device)
    model.eval()
    
    # Original predictions
    with torch.no_grad():
        clean_outputs = model(x)
        clean_preds = clean_outputs.argmax(dim=1)
        clean_acc = clean_preds.eq(y).float().mean().item()
    
    results = {
        'clean_accuracy': clean_acc * 100,
        'attacks': {}
    }
    
    # FGSM
    fgsm = FGSMAttack(epsilon=epsilon)
    fgsm_adv = fgsm.generate(model, x, y)
    with torch.no_grad():
        fgsm_preds = model(fgsm_adv).argmax(dim=1)
        fgsm_acc = fgsm_preds.eq(y).float().mean().item()
    results['attacks']['FGSM'] = {
        'accuracy': fgsm_acc * 100,
        'success_rate': (1 - fgsm_acc) * 100
    }
    
    # I-FGSM
    ifgsm = IterativeFGSM(epsilon=epsilon, alpha=epsilon/10, num_iter=10)
    ifgsm_adv = ifgsm.generate(model, x, y)
    with torch.no_grad():
        ifgsm_preds = model(ifgsm_adv).argmax(dim=1)
        ifgsm_acc = ifgsm_preds.eq(y).float().mean().item()
    results['attacks']['I-FGSM'] = {
        'accuracy': ifgsm_acc * 100,
        'success_rate': (1 - ifgsm_acc) * 100
    }
    
    # PGD
    pgd = PGDAttack(epsilon=epsilon, alpha=epsilon/25, num_steps=40)
    pgd_adv = pgd.generate(model, x, y)
    with torch.no_grad():
        pgd_preds = model(pgd_adv).argmax(dim=1)
        pgd_acc = pgd_preds.eq(y).float().mean().item()
    results['attacks']['PGD'] = {
        'accuracy': pgd_acc * 100,
        'success_rate': (1 - pgd_acc) * 100
    }
    
    return results