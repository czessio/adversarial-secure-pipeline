# Bias analysis for dataset and model evaluation

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from ..attacks.fgsm import FGSMAttack
from ..attacks.pgd import PGDAttack
from ..preprocessing.lighting_correction import visualise_lighting_correction


class BiasAnalyser:
    """Analyse bias in dataset and model predictions."""
    
    def __init__(self, config: Dict[str, Any], class_names: List[str]):
        """
        Initialise bias analyser.
        
        Args:
            config: Configuration dictionary
            class_names: List of class names
        """
        self.config = config
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.device = torch.device(config['hardware']['device'])
        
        # Initialise attacks for robustness analysis
        self.fgsm = FGSMAttack(epsilon=config['adversarial']['attacks']['fgsm']['epsilon'])
        self.pgd = PGDAttack(
            epsilon=config['adversarial']['attacks']['pgd']['epsilon'],
            alpha=config['adversarial']['attacks']['pgd']['alpha'],
            num_steps=config['adversarial']['attacks']['pgd']['num_steps']
        )
    
    def analyse_class_distribution(self, data_loader: DataLoader) -> Dict[str, Any]:
        """
        Analyse class distribution in dataset.
        
        Args:
            data_loader: Data loader to analyse
        
        Returns:
            Distribution statistics
        """
        class_counts = torch.zeros(self.num_classes)
        
        for _, labels in data_loader:
            for label in labels:
                class_counts[label] += 1
        
        total_samples = class_counts.sum().item()
        class_proportions = class_counts / total_samples
        
        # Calculate imbalance metrics
        max_prop = class_proportions.max().item()
        min_prop = class_proportions.min().item()
        imbalance_ratio = max_prop / min_prop
        
        # Entropy as measure of balance
        entropy = -torch.sum(class_proportions * torch.log(class_proportions + 1e-10)).item()
        max_entropy = np.log(self.num_classes)
        normalised_entropy = entropy / max_entropy
        
        return {
            'class_counts': class_counts.numpy(),
            'class_proportions': class_proportions.numpy(),
            'imbalance_ratio': imbalance_ratio,
            'normalised_entropy': normalised_entropy,
            'most_common_class': self.class_names[class_counts.argmax()],
            'least_common_class': self.class_names[class_counts.argmin()]
        }
    
    def analyse_model_bias(
        self,
        model: nn.Module,
        test_loader: DataLoader
    ) -> Dict[str, Any]:
        """
        Analyse model bias across classes.
        
        Args:
            model: Model to analyse
            test_loader: Test data loader
        
        Returns:
            Bias analysis results
        """
        model.eval()
        model.to(self.device)
        
        # Initialise metrics
        class_correct = torch.zeros(self.num_classes)
        class_total = torch.zeros(self.num_classes)
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Analysing model bias"):
                data, target = data.to(self.device), target.to(self.device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(target.cpu().numpy())
                
                # Update per-class statistics
                for i in range(target.size(0)):
                    label = target[i]
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1
        
        # Calculate per-class accuracy
        class_accuracy = (class_correct / (class_total + 1e-10)).numpy()
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        # Calculate bias metrics
        accuracy_variance = np.var(class_accuracy)
        accuracy_std = np.std(class_accuracy)
        worst_class_idx = np.argmin(class_accuracy)
        best_class_idx = np.argmax(class_accuracy)
        
        return {
            'class_accuracy': class_accuracy,
            'overall_accuracy': (class_correct.sum() / class_total.sum()).item(),
            'accuracy_variance': accuracy_variance,
            'accuracy_std': accuracy_std,
            'worst_class': self.class_names[worst_class_idx],
            'worst_class_accuracy': class_accuracy[worst_class_idx],
            'best_class': self.class_names[best_class_idx],
            'best_class_accuracy': class_accuracy[best_class_idx],
            'confusion_matrix': conf_matrix
        }
    
    def analyse_robustness_bias(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        attack_type: str = 'pgd'
    ) -> Dict[str, Any]:
        """
        Analyse robustness bias across classes.
        
        Args:
            model: Model to analyse
            test_loader: Test data loader
            attack_type: Type of attack ('fgsm' or 'pgd')
        
        Returns:
            Robustness bias analysis
        """
        model.eval()
        model.to(self.device)
        
        attack = self.fgsm if attack_type == 'fgsm' else self.pgd
        
        # Initialise metrics
        class_robust_correct = torch.zeros(self.num_classes)
        class_total = torch.zeros(self.num_classes)
        
        for data, target in tqdm(test_loader, desc=f"Analysing {attack_type.upper()} robustness"):
            data, target = data.to(self.device), target.to(self.device)
            
            # Generate adversarial examples
            adv_data = attack.generate(model, data, target)
            
            # Evaluate on adversarial examples
            with torch.no_grad():
                outputs = model(adv_data)
                _, predicted = outputs.max(1)
                
                # Update per-class statistics
                for i in range(target.size(0)):
                    label = target[i]
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_robust_correct[label] += 1
        
        # Calculate per-class robustness
        class_robustness = (class_robust_correct / (class_total + 1e-10)).numpy()
        
        # Calculate robustness bias metrics
        robustness_variance = np.var(class_robustness)
        robustness_std = np.std(class_robustness)
        least_robust_idx = np.argmin(class_robustness)
        most_robust_idx = np.argmax(class_robustness)
        
        return {
            'class_robustness': class_robustness,
            'overall_robustness': (class_robust_correct.sum() / class_total.sum()).item(),
            'robustness_variance': robustness_variance,
            'robustness_std': robustness_std,
            'least_robust_class': self.class_names[least_robust_idx],
            'least_robust_accuracy': class_robustness[least_robust_idx],
            'most_robust_class': self.class_names[most_robust_idx],
            'most_robust_accuracy': class_robustness[most_robust_idx],
            'attack_type': attack_type
        }
    
    def analyse_lighting_sensitivity(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Analyse model sensitivity to lighting variations.
        
        Args:
            model: Model to analyse
            test_loader: Test data loader
            num_samples: Number of samples to analyse
        
        Returns:
            Lighting sensitivity analysis
        """
        model.eval()
        model.to(self.device)
        
        sensitivity_scores = []
        class_sensitivity = {class_name: [] for class_name in self.class_names}
        
        sample_count = 0
        for data, target in test_loader:
            if sample_count >= num_samples:
                break
            
            data, target = data.to(self.device), target.to(self.device)
            
            for i in range(data.size(0)):
                if sample_count >= num_samples:
                    break
                
                image = data[i].cpu().numpy().transpose(1, 2, 0)
                label = target[i].item()
                
                # Apply lighting corrections
                corrected_images = visualise_lighting_correction(image, self.config)
                
                # Evaluate model on each variant
                predictions = {}
                with torch.no_grad():
                    for variant_name, variant_image in corrected_images.items():
                        if isinstance(variant_image, np.ndarray):
                            variant_tensor = torch.from_numpy(
                                variant_image.transpose(2, 0, 1)
                            ).unsqueeze(0).float().to(self.device)
                            
                            output = model(variant_tensor)
                            pred = output.argmax(1).item()
                            predictions[variant_name] = pred
                
                # Calculate sensitivity (proportion of variants with different predictions)
                original_pred = predictions.get('original', label)
                different_preds = sum(1 for p in predictions.values() if p != original_pred)
                sensitivity = different_preds / len(predictions)
                
                sensitivity_scores.append(sensitivity)
                class_sensitivity[self.class_names[label]].append(sensitivity)
                sample_count += 1
        
        # Calculate statistics
        overall_sensitivity = np.mean(sensitivity_scores)
        class_avg_sensitivity = {
            class_name: np.mean(scores) if scores else 0
            for class_name, scores in class_sensitivity.items()
        }
        
        most_sensitive_class = max(class_avg_sensitivity, key=class_avg_sensitivity.get)
        least_sensitive_class = min(class_avg_sensitivity, key=class_avg_sensitivity.get)
        
        return {
            'overall_sensitivity': overall_sensitivity,
            'class_sensitivity': class_avg_sensitivity,
            'most_sensitive_class': most_sensitive_class,
            'least_sensitive_class': least_sensitive_class,
            'sensitivity_variance': np.var(list(class_avg_sensitivity.values()))
        }
    
    def generate_bias_report(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader
    ) -> pd.DataFrame:
        """
        Generate comprehensive bias report.
        
        Args:
            model: Model to analyse
            train_loader: Training data loader
            test_loader: Test data loader
        
        Returns:
            Bias report as DataFrame
        """
        print("Generating comprehensive bias analysis report...")
        
        # Analyse dataset distribution
        print("\n1. Analysing dataset distribution...")
        train_dist = self.analyse_class_distribution(train_loader)
        test_dist = self.analyse_class_distribution(test_loader)
        
        # Analyse model bias
        print("\n2. Analysing model prediction bias...")
        model_bias = self.analyse_model_bias(model, test_loader)
        
        # Analyse robustness bias
        print("\n3. Analysing adversarial robustness bias...")
        fgsm_robustness = self.analyse_robustness_bias(model, test_loader, 'fgsm')
        pgd_robustness = self.analyse_robustness_bias(model, test_loader, 'pgd')
        
        # Analyse lighting sensitivity
        print("\n4. Analysing lighting sensitivity...")
        lighting_sensitivity = self.analyse_lighting_sensitivity(model, test_loader)
        
        # Compile report
        report_data = []
        
        for i, class_name in enumerate(self.class_names):
            report_data.append({
                'Class': class_name,
                'Train Proportion': train_dist['class_proportions'][i],
                'Test Proportion': test_dist['class_proportions'][i],
                'Accuracy': model_bias['class_accuracy'][i],
                'FGSM Robustness': fgsm_robustness['class_robustness'][i],
                'PGD Robustness': pgd_robustness['class_robustness'][i],
                'Lighting Sensitivity': lighting_sensitivity['class_sensitivity'].get(class_name, 0)
            })
        
        report_df = pd.DataFrame(report_data)
        
        # Add summary statistics
        summary_stats = pd.DataFrame([{
            'Class': 'Overall',
            'Train Proportion': 1.0,
            'Test Proportion': 1.0,
            'Accuracy': model_bias['overall_accuracy'],
            'FGSM Robustness': fgsm_robustness['overall_robustness'],
            'PGD Robustness': pgd_robustness['overall_robustness'],
            'Lighting Sensitivity': lighting_sensitivity['overall_sensitivity']
        }, {
            'Class': 'Std Dev',
            'Train Proportion': np.std(train_dist['class_proportions']),
            'Test Proportion': np.std(test_dist['class_proportions']),
            'Accuracy': model_bias['accuracy_std'],
            'FGSM Robustness': fgsm_robustness['robustness_std'],
            'PGD Robustness': pgd_robustness['robustness_std'],
            'Lighting Sensitivity': np.std(list(lighting_sensitivity['class_sensitivity'].values()))
        }])
        
        report_df = pd.concat([report_df, summary_stats], ignore_index=True)
        
        return report_df
    
    def identify_bias_patterns(self, bias_report: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify patterns in bias analysis.
        
        Args:
            bias_report: Bias report DataFrame
        
        Returns:
            Identified bias patterns
        """
        # Exclude summary rows
        class_data = bias_report[~bias_report['Class'].isin(['Overall', 'Std Dev'])]
        
        patterns = {
            'underrepresented_vulnerable': [],
            'overrepresented_vulnerable': [],
            'robust_but_inaccurate': [],
            'accurate_but_fragile': [],
            'lighting_vulnerable': []
        }
        
        # Identify underrepresented but vulnerable classes
        median_proportion = class_data['Train Proportion'].median()
        median_robustness = class_data['PGD Robustness'].median()
        
        for _, row in class_data.iterrows():
            # Underrepresented and vulnerable
            if (row['Train Proportion'] < median_proportion and 
                row['PGD Robustness'] < median_robustness):
                patterns['underrepresented_vulnerable'].append(row['Class'])
            
            # Overrepresented but still vulnerable
            if (row['Train Proportion'] > median_proportion and 
                row['PGD Robustness'] < median_robustness):
                patterns['overrepresented_vulnerable'].append(row['Class'])
            
            # Robust but inaccurate
            if (row['PGD Robustness'] > median_robustness and 
                row['Accuracy'] < class_data['Accuracy'].median()):
                patterns['robust_but_inaccurate'].append(row['Class'])
            
            # Accurate but fragile
            if (row['Accuracy'] > class_data['Accuracy'].median() and 
                row['PGD Robustness'] < median_robustness):
                patterns['accurate_but_fragile'].append(row['Class'])
            
            # Lighting vulnerable
            if row['Lighting Sensitivity'] > class_data['Lighting Sensitivity'].median():
                patterns['lighting_vulnerable'].append(row['Class'])
        
        return patterns