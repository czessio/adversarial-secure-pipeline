# Fast Gradient Sign Method (FGSM) adversarial attack implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class FGSMAttack:
    """Fast Gradient Sign Method attack."""
    
    def __init__(self, epsilon: float = 0.03, targeted: bool = False, clip_min: float = 0.0, clip_max: float = 1.0):
        """
        Initialise FGSM attack.
        
        Args:
            epsilon: Maximum perturbation magnitude
            targeted: Whether to perform targeted attack
            clip_min: Minimum pixel value
            clip_max: Maximum pixel value
        """
        self.epsilon = epsilon
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
        Generate adversarial examples using FGSM.
        
        Args:
            model: Target model
            x: Input images
            y: True labels
            target: Target labels for targeted attack
        
        Returns:
            Adversarial examples
        """
        # Ensure model is in eval mode
        model.eval()
        
        # Clone input to avoid modifying original
        x_adv = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = model(x_adv)
        
        # Calculate loss
        if self.targeted and target is not None:
            # For targeted attack, minimise loss w.r.t. target class
            loss = F.cross_entropy(outputs, target)
            loss = -loss  # Minimise instead of maximise
        else:
            # For untargeted attack, maximise loss w.r.t. true class
            loss = F.cross_entropy(outputs, y)
        
        # Calculate gradients
        loss.backward()
        
        # Get gradient sign
        grad_sign = x_adv.grad.sign()
        
        # Create adversarial examples
        x_adv = x_adv + self.epsilon * grad_sign
        
        # Clip to valid range
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


class IterativeFGSM(FGSMAttack):
    """Iterative FGSM (I-FGSM) attack."""
    
    def __init__(
        self,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        num_iter: int = 10,
        targeted: bool = False,
        clip_min: float = 0.0,
        clip_max: float = 1.0
    ):
        """
        Initialise I-FGSM attack.
        
        Args:
            epsilon: Maximum perturbation magnitude
            alpha: Step size for each iteration
            num_iter: Number of iterations
            targeted: Whether to perform targeted attack
            clip_min: Minimum pixel value
            clip_max: Maximum pixel value
        """
        super().__init__(epsilon, targeted, clip_min, clip_max)
        self.alpha = alpha
        self.num_iter = num_iter
    
    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate adversarial examples using I-FGSM."""
        model.eval()
        
        # Start from original input
        x_adv = x.clone().detach()
        
        for i in range(self.num_iter):
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
            
            # Update adversarial examples
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv + self.alpha * grad_sign
            
            # Project back to epsilon ball
            delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = x + delta
            
            # Clip to valid range
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max).detach()
        
        return x_adv


def evaluate_fgsm_robustness(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    epsilon_values: list,
    device: torch.device = torch.device('cpu')
) -> dict:
    """
    Evaluate model robustness against FGSM at different epsilon values.
    
    Args:
        model: Model to evaluate
        data_loader: Test data loader
        epsilon_values: List of epsilon values to test
        device: Device to use
    
    Returns:
        Dictionary with robustness metrics
    """
    results = {
        'epsilon': epsilon_values,
        'accuracy': [],
        'attack_success_rate': []
    }
    
    for eps in epsilon_values:
        fgsm = FGSMAttack(epsilon=eps)
        
        correct = 0
        successful_attacks = 0
        total = 0
        
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Generate adversarial examples
            adv_data = fgsm.generate(model, data, target)
            
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