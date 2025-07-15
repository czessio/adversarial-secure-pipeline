# Script to analyse bias in dataset and model

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.data_loader import load_config, create_data_loaders, get_class_names
from src.models.base_classifier import create_model
from src.analysis.bias_analysis import BiasAnalyser
from src.analysis.visualisation import setup_plotting_style, plot_bias_analysis


def main(args):
    """Main bias analysis function."""
    # Setup plotting style
    setup_plotting_style()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\nPreparing data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Get class names
    class_names = get_class_names(config['data']['dataset_name'])
    
    # Load model if provided
    model = None
    if args.model_path:
        print(f"\nLoading model from: {args.model_path}")
        model = create_model(config)
        
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialise bias analyser
    analyser = BiasAnalyser(config, class_names)
    
    # 1. Dataset bias analysis
    print("\n" + "="*50)
    print("DATASET BIAS ANALYSIS")
    print("="*50)
    
    print("\nAnalysing training set distribution...")
    train_dist = analyser.analyse_class_distribution(train_loader)
    
    print("\nAnalysing test set distribution...")
    test_dist = analyser.analyse_class_distribution(test_loader)
    
    # Plot class distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training distribution
    ax1.bar(class_names, train_dist['class_proportions'])
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Proportion')
    ax1.set_title('Training Set Class Distribution')
    ax1.tick_params(axis='x', rotation=45)
    
    # Test distribution
    ax2.bar(class_names, test_dist['class_proportions'])
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Proportion')
    ax2.set_title('Test Set Class Distribution')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'class_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print distribution statistics
    print(f"\nTraining set:")
    print(f"  - Imbalance ratio: {train_dist['imbalance_ratio']:.2f}")
    print(f"  - Normalised entropy: {train_dist['normalised_entropy']:.2f}")
    print(f"  - Most common class: {train_dist['most_common_class']}")
    print(f"  - Least common class: {train_dist['least_common_class']}")
    
    print(f"\nTest set:")
    print(f"  - Imbalance ratio: {test_dist['imbalance_ratio']:.2f}")
    print(f"  - Normalised entropy: {test_dist['normalised_entropy']:.2f}")
    print(f"  - Most common class: {test_dist['most_common_class']}")
    print(f"  - Least common class: {test_dist['least_common_class']}")
    
    # 2. Model bias analysis (if model provided)
    if model is not None:
        print("\n" + "="*50)
        print("MODEL BIAS ANALYSIS")
        print("="*50)
        
        # Analyse model predictions
        print("\nAnalysing model bias on test set...")
        model_bias = analyser.analyse_model_bias(model, test_loader)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            model_bias['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            square=True
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(results_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot per-class accuracy
        plt.figure(figsize=(10, 6))
        plt.bar(class_names, model_bias['class_accuracy'])
        plt.axhline(y=model_bias['overall_accuracy'], color='r', linestyle='--', 
                   label=f"Overall: {model_bias['overall_accuracy']:.2f}%")
        plt.xlabel('Class')
        plt.ylabel('Accuracy (%)')
        plt.title('Per-Class Accuracy')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(results_dir / 'per_class_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print model bias statistics
        print(f"\nOverall accuracy: {model_bias['overall_accuracy']:.2f}%")
        print(f"Accuracy variance: {model_bias['accuracy_variance']:.4f}")
        print(f"Accuracy std dev: {model_bias['accuracy_std']:.2f}")
        print(f"Best performing class: {model_bias['best_class']} ({model_bias['best_class_accuracy']:.2f}%)")
        print(f"Worst performing class: {model_bias['worst_class']} ({model_bias['worst_class_accuracy']:.2f}%)")
        
        # 3. Robustness bias analysis
        if args.analyse_robustness:
            print("\n" + "="*50)
            print("ROBUSTNESS BIAS ANALYSIS")
            print("="*50)
            
            # FGSM robustness
            print("\nAnalysing FGSM robustness bias...")
            fgsm_bias = analyser.analyse_robustness_bias(model, test_loader, 'fgsm')
            
            # PGD robustness
            print("\nAnalysing PGD robustness bias...")
            pgd_bias = analyser.analyse_robustness_bias(model, test_loader, 'pgd')
            
            # Plot robustness comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # FGSM robustness
            ax1.bar(class_names, fgsm_bias['class_robustness'])
            ax1.axhline(y=fgsm_bias['overall_robustness'], color='r', linestyle='--',
                       label=f"Overall: {fgsm_bias['overall_robustness']:.2f}%")
            ax1.set_xlabel('Class')
            ax1.set_ylabel('Robustness (%)')
            ax1.set_title('FGSM Robustness by Class')
            ax1.legend()
            ax1.tick_params(axis='x', rotation=45)
            
            # PGD robustness
            ax2.bar(class_names, pgd_bias['class_robustness'])
            ax2.axhline(y=pgd_bias['overall_robustness'], color='r', linestyle='--',
                       label=f"Overall: {pgd_bias['overall_robustness']:.2f}%")
            ax2.set_xlabel('Class')
            ax2.set_ylabel('Robustness (%)')
            ax2.set_title('PGD Robustness by Class')
            ax2.legend()
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(results_dir / 'robustness_bias.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Print robustness statistics
            print(f"\nFGSM Robustness:")
            print(f"  - Overall: {fgsm_bias['overall_robustness']:.2f}%")
            print(f"  - Variance: {fgsm_bias['robustness_variance']:.4f}")
            print(f"  - Most robust: {fgsm_bias['most_robust_class']} ({fgsm_bias['most_robust_accuracy']:.2f}%)")
            print(f"  - Least robust: {fgsm_bias['least_robust_class']} ({fgsm_bias['least_robust_accuracy']:.2f}%)")
            
            print(f"\nPGD Robustness:")
            print(f"  - Overall: {pgd_bias['overall_robustness']:.2f}%")
            print(f"  - Variance: {pgd_bias['robustness_variance']:.4f}")
            print(f"  - Most robust: {pgd_bias['most_robust_class']} ({pgd_bias['most_robust_accuracy']:.2f}%)")
            print(f"  - Least robust: {pgd_bias['least_robust_class']} ({pgd_bias['least_robust_accuracy']:.2f}%)")
        
        # 4. Lighting sensitivity analysis
        if args.analyse_lighting:
            print("\n" + "="*50)
            print("LIGHTING SENSITIVITY ANALYSIS")
            print("="*50)
            
            print("\nAnalysing lighting sensitivity...")
            lighting_bias = analyser.analyse_lighting_sensitivity(
                model, test_loader, num_samples=args.num_lighting_samples
            )
            
            # Plot lighting sensitivity
            plt.figure(figsize=(10, 6))
            sensitivity_values = [lighting_bias['class_sensitivity'][c] for c in class_names]
            plt.bar(class_names, sensitivity_values)
            plt.axhline(y=lighting_bias['overall_sensitivity'], color='r', linestyle='--',
                       label=f"Overall: {lighting_bias['overall_sensitivity']:.3f}")
            plt.xlabel('Class')
            plt.ylabel('Lighting Sensitivity')
            plt.title('Lighting Sensitivity by Class')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(results_dir / 'lighting_sensitivity.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\nOverall sensitivity: {lighting_bias['overall_sensitivity']:.3f}")
            print(f"Most sensitive: {lighting_bias['most_sensitive_class']}")
            print(f"Least sensitive: {lighting_bias['least_sensitive_class']}")
        
        # 5. Generate comprehensive bias report
        print("\n" + "="*50)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*50)
        
        bias_report = analyser.generate_bias_report(model, train_loader, test_loader)
        
        # Save report
        report_path = results_dir / 'bias_analysis_report.csv'
        bias_report.to_csv(report_path, index=False)
        print(f"\nBias report saved to: {report_path}")
        
        # Create visualisation
        plot_bias_analysis(bias_report, str(results_dir / 'bias_analysis_dashboard.html'))
        
        # Identify bias patterns
        patterns = analyser.identify_bias_patterns(bias_report)
        
        print("\nIdentified Bias Patterns:")
        for pattern_name, classes in patterns.items():
            if classes:
                print(f"  - {pattern_name.replace('_', ' ').title()}: {', '.join(classes)}")
    
    print("\n" + "="*50)
    print("BIAS ANALYSIS COMPLETE")
    print("="*50)
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse bias in dataset and model")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained model checkpoint (optional)")
    parser.add_argument("--results-dir", type=str, default="results/bias_analysis",
                        help="Directory to save results")
    parser.add_argument("--analyse-robustness", action="store_true",
                        help="Analyse adversarial robustness bias")
    parser.add_argument("--analyse-lighting", action="store_true",
                        help="Analyse lighting sensitivity")
    parser.add_argument("--num-lighting-samples", type=int, default=100,
                        help="Number of samples for lighting analysis")
    
    args = parser.parse_args()
    main(args)