# Metrics utilities for model evaluation

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, average_precision_score
)
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd


class MetricsTracker:
    """Track and compute various metrics during training and evaluation."""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialise metrics tracker.
        
        Args:
            num_classes: Number of classes
            class_names: Optional list of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.losses = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None,
        loss: Optional[float] = None
    ):
        """
        Update metrics with batch results.
        
        Args:
            predictions: Predicted labels
            targets: True labels
            probabilities: Prediction probabilities
            loss: Batch loss
        """
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        
        if probabilities is not None:
            self.probabilities.extend(probabilities.cpu().numpy())
        
        if loss is not None:
            self.losses.append(loss)
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute all metrics."""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics with support
        per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(targets, predictions)
        
        metrics = {
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'average_loss': np.mean(self.losses) if self.losses else 0,
            'confusion_matrix': conf_matrix,
            'per_class_metrics': {
                self.class_names[i]: {
                    'precision': per_class_precision[i] * 100 if i < len(per_class_precision) else 0,
                    'recall': per_class_recall[i] * 100 if i < len(per_class_recall) else 0,
                    'f1_score': per_class_f1[i] * 100 if i < len(per_class_f1) else 0,
                    'support': int(per_class_support[i]) if per_class_support is not None and i < len(per_class_support) else 0
                }
                for i in range(self.num_classes)
            }
        }
        
        # Add AUC if probabilities available
        if self.probabilities:
            probabilities = np.array(self.probabilities)
            if self.num_classes == 2:
                # Binary classification
                auc = roc_auc_score(targets, probabilities[:, 1])
                metrics['auc'] = auc
            else:
                # Multi-class: one-vs-rest AUC
                try:
                    # Convert to one-hot encoding
                    targets_onehot = np.eye(self.num_classes)[targets]
                    auc = roc_auc_score(targets_onehot, probabilities, average='weighted')
                    metrics['auc'] = auc
                except:
                    pass
        
        return metrics
    
    def get_classification_report(self) -> str:
        """Generate detailed classification report."""
        metrics = self.compute_metrics()
        
        report = f"\nClassification Report\n{'='*50}\n"
        report += f"Overall Accuracy: {metrics['accuracy']:.2f}%\n"
        report += f"Weighted Precision: {metrics['precision']:.2f}%\n"
        report += f"Weighted Recall: {metrics['recall']:.2f}%\n"
        report += f"Weighted F1-Score: {metrics['f1_score']:.2f}%\n"
        
        if 'auc' in metrics:
            report += f"AUC Score: {metrics['auc']:.4f}\n"
        
        report += f"\nPer-Class Metrics:\n{'-'*50}\n"
        report += f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n"
        report += f"{'-'*61}\n"
        
        for class_name, class_metrics in metrics['per_class_metrics'].items():
            report += f"{class_name:<15} "
            report += f"{class_metrics['precision']:<12.2f} "
            report += f"{class_metrics['recall']:<12.2f} "
            report += f"{class_metrics['f1_score']:<12.2f} "
            report += f"{class_metrics['support']:<10}\n"
        
        return report


def calculate_robustness_metrics(
    clean_accuracy: float,
    adversarial_accuracy: float,
    attack_success_rate: float
) -> Dict[str, float]:
    """
    Calculate robustness-specific metrics.
    
    Args:
        clean_accuracy: Accuracy on clean examples
        adversarial_accuracy: Accuracy on adversarial examples
        attack_success_rate: Successful attack rate
    
    Returns:
        Robustness metrics
    """
    robustness_gap = clean_accuracy - adversarial_accuracy
    relative_robustness = adversarial_accuracy / clean_accuracy if clean_accuracy > 0 else 0
    
    return {
        'robustness_gap': robustness_gap,
        'relative_robustness': relative_robustness * 100,
        'attack_success_rate': attack_success_rate,
        'defense_success_rate': 100 - attack_success_rate
    }


def calculate_efficiency_metrics(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device = torch.device('cpu'),
    num_runs: int = 100
) -> Dict[str, Any]:
    """
    Calculate model efficiency metrics.
    
    Args:
        model: Model to evaluate
        input_shape: Input tensor shape
        device: Device to use
        num_runs: Number of inference runs
    
    Returns:
        Efficiency metrics
    """
    import time
    
    model.eval()
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Measure inference time
    dummy_input = torch.randn(1, *input_shape[1:]).to(device)
    
    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Time inference
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    total_time = time.time() - start_time
    
    # Calculate FLOPs (simplified estimation)
    from thop import profile
    try:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    except:
        flops = 0
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'inference_time_ms': (total_time / num_runs) * 1000,
        'throughput_fps': num_runs / total_time,
        'flops': flops,
        'flops_formatted': f"{flops/1e9:.2f}G" if flops > 0 else "N/A"
    }


def calculate_calibration_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 10
) -> Dict[str, float]:
    """
    Calculate model calibration metrics.
    
    Args:
        probabilities: Predicted probabilities
        targets: True labels
        num_bins: Number of bins for calibration
    
    Returns:
        Calibration metrics
    """
    # Get predicted classes and confidence
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    accuracies = predictions == targets
    
    # Calculate ECE (Expected Calibration Error)
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    mce = 0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += prop_in_bin * calibration_error
            mce = max(mce, calibration_error)
    
    # Calculate Brier score
    num_classes = probabilities.shape[1]
    targets_onehot = np.eye(num_classes)[targets]
    brier_score = np.mean(np.sum((probabilities - targets_onehot) ** 2, axis=1))
    
    return {
        'expected_calibration_error': ece,
        'maximum_calibration_error': mce,
        'brier_score': brier_score,
        'mean_confidence': np.mean(confidences),
        'mean_accuracy': np.mean(accuracies)
    }


class RobustnessEvaluator:
    """Comprehensive robustness evaluation."""
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        """
        Initialise robustness evaluator.
        
        Args:
            model: Model to evaluate
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        
        # Auto-detect device availability
        device_config = config['hardware']['device']
        if device_config == 'cuda' and not torch.cuda.is_available():
            print("  RobustnessEvaluator: CUDA not available, using CPU")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_config if device_config != 'cuda' or torch.cuda.is_available() else 'cpu')
    
    
    def evaluate_comprehensive(
        self,
        test_loader: torch.utils.data.DataLoader,
        attacks: List[Any]
    ) -> pd.DataFrame:
        """
        Perform comprehensive robustness evaluation.
        
        Args:
            test_loader: Test data loader
            attacks: List of attack instances
        
        Returns:
            Results DataFrame
        """
        results = []
        
        # Evaluate clean accuracy
        clean_metrics = self._evaluate_clean(test_loader)
        results.append({
            'attack': 'None',
            'epsilon': 0.0,
            **clean_metrics
        })
        
        # Evaluate against each attack
        for attack in attacks:
            attack_name = attack.__class__.__name__
            
            if hasattr(attack, 'epsilon'):
                # Single epsilon
                attack_metrics = self._evaluate_attack(test_loader, attack)
                results.append({
                    'attack': attack_name,
                    'epsilon': attack.epsilon,
                    **attack_metrics
                })
            else:
                # Multiple epsilon values
                for eps in [0.01, 0.03, 0.05, 0.1]:
                    attack.epsilon = eps
                    attack_metrics = self._evaluate_attack(test_loader, attack)
                    results.append({
                        'attack': attack_name,
                        'epsilon': eps,
                        **attack_metrics
                    })
        
        return pd.DataFrame(results)
    
    
    
    
    def _evaluate_clean(self, test_loader) -> Dict[str, float]:
        """Evaluate on clean examples."""
        tracker = MetricsTracker(self.config['model']['num_classes'])
        self.model.eval()
        
        for data, target in test_loader:
            # Handle quantized models on CPU
            if hasattr(self.model, 'qconfig') or any(
                isinstance(m, torch.nn.quantized.Linear)
                for m in self.model.modules()
            ):
                self.device = torch.device('cpu')
                data = data.cpu()
                target = target.cpu()
            else:
                data, target = data.to(self.device), target.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(data)
                predictions = outputs.argmax(dim=1)
                probabilities = torch.softmax(outputs, dim=1)
            
            tracker.update(predictions, target, probabilities)
        
        metrics = tracker.compute_metrics()
        return {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        }


    
 
    
    def _evaluate_attack(self, test_loader, attack) -> Dict[str, float]:
        """Evaluate against specific attack."""
        tracker = MetricsTracker(self.config['model']['num_classes'])
        self.model.eval()
        
        for data, target in test_loader:
            # Handle quantized models on CPU
            if hasattr(self.model, 'qconfig') or any(
                isinstance(m, torch.nn.quantized.Linear)
                for m in self.model.modules()
            ):
                self.device = torch.device('cpu')
                data = data.cpu()
                target = target.cpu()
            else:
                data, target = data.to(self.device), target.to(self.device)
            
            # Generate adversarial examples
            adv_data = attack.generate(self.model, data, target)
            
            with torch.no_grad():
                outputs = self.model(adv_data)
                predictions = outputs.argmax(dim=1)
                probabilities = torch.softmax(outputs, dim=1)
            
            tracker.update(predictions, target, probabilities)
        
        metrics = tracker.compute_metrics()
        return {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        }