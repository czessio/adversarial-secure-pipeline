# Script to evaluate model robustness against adversarial attacks

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from src.utils.data_loader import load_config, create_data_loaders
from src.models.base_classifier import create_model
from src.attacks.fgsm import evaluate_fgsm_robustness
from src.attacks.pgd import evaluate_pgd_robustness, compare_attacks
from src.privacy.homomorphic_encryption import demonstrate_private_inference


def evaluate_clean_accuracy(model, test_loader, device):
    """Evaluate model accuracy on clean examples."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def plot_robustness_curves(results, save_path):
    """Plot robustness curves for different attacks."""
    plt.figure(figsize=(10, 6))
    
    # Plot FGSM results
    if 'fgsm' in results:
        plt.plot(results['fgsm']['epsilon'], results['fgsm']['accuracy'], 
                'b-o', label='FGSM', linewidth=2, markersize=8)
    
    # Plot PGD results
    if 'pgd' in results:
        plt.plot(results['pgd']['epsilon'], results['pgd']['accuracy'], 
                'r-s', label='PGD', linewidth=2, markersize=8)
    
    plt.xlabel('Epsilon (ε)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Robustness Against Adversarial Attacks', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Robustness plot saved to: {save_path}")


def main(args):
    """Main evaluation function."""
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\nPreparing data loaders...")
    _, _, test_loader = create_data_loaders(config)
    
    # Create and load model
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
    
    # Evaluate clean accuracy
    print("\nEvaluating clean accuracy...")
    clean_accuracy = evaluate_clean_accuracy(model, test_loader, device)
    print(f"Clean accuracy: {clean_accuracy:.2f}%")
    
    results = {
        'clean_accuracy': clean_accuracy,
        'model_path': str(args.model_path),
        'dataset': config['data']['dataset_name']
    }
    
    # Evaluate FGSM robustness
    if not args.skip_fgsm:
        print("\nEvaluating FGSM robustness...")
        epsilon_values = np.linspace(0, args.max_epsilon, args.num_epsilon_steps).tolist()
        fgsm_results = evaluate_fgsm_robustness(model, test_loader, epsilon_values, device)
        results['fgsm'] = fgsm_results
    
    # Evaluate PGD robustness
    if not args.skip_pgd:
        print("\nEvaluating PGD robustness...")
        epsilon_values = np.linspace(0, args.max_epsilon, args.num_epsilon_steps).tolist()
        pgd_results = evaluate_pgd_robustness(
            model, test_loader, epsilon_values, device, 
            num_steps=args.pgd_steps, alpha_ratio=args.pgd_alpha_ratio
        )
        results['pgd'] = pgd_results
    
    # Compare attacks on sample batch
    if args.compare_attacks:
        print("\nComparing attacks on sample batch...")
        sample_data, sample_labels = next(iter(test_loader))
        comparison = compare_attacks(
            model, sample_data, sample_labels, 
            epsilon=config['adversarial']['attacks']['pgd']['epsilon'], 
            device=device
        )
        results['attack_comparison'] = comparison
        
        print("\nAttack Comparison:")
        print(f"Clean accuracy: {comparison['clean_accuracy']:.2f}%")
        for attack_name, attack_results in comparison['attacks'].items():
            print(f"{attack_name}: Accuracy={attack_results['accuracy']:.2f}%, "
                  f"Success Rate={attack_results['success_rate']:.2f}%")
    
    # Test homomorphic encryption if enabled
    if args.test_encryption and config['privacy']['homomorphic_encryption']['enable']:
        print("\nTesting homomorphic encryption...")
        sample_data, _ = next(iter(test_loader))
        sample_input = sample_data[0:1]  # Single sample
        
        encryption_results = demonstrate_private_inference(model, sample_input, config)
        results['encryption_test'] = {
            'success': encryption_results['success'],
            'absolute_error': encryption_results['absolute_error']
        }
        
        print(f"Encryption test success: {encryption_results['success']}")
        print(f"Absolute error: {encryption_results['absolute_error']:.6f}")
    
    # Save results
    results_path = results_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Plot robustness curves
    if not args.skip_plots:
        plot_path = results_dir / "robustness_curves.png"
        plot_results = {}
        if 'fgsm' in results:
            plot_results['fgsm'] = results['fgsm']
        if 'pgd' in results:
            plot_results['pgd'] = results['pgd']
        
        if plot_results:
            plot_robustness_curves(plot_results, plot_path)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {config['data']['dataset_name']}")
    print(f"Clean Accuracy: {clean_accuracy:.2f}%")
    
    if 'fgsm' in results:
        eps = config['adversarial']['attacks']['fgsm']['epsilon']
        idx = np.argmin(np.abs(np.array(results['fgsm']['epsilon']) - eps))
        print(f"FGSM Accuracy (ε={eps}): {results['fgsm']['accuracy'][idx]:.2f}%")
    
    if 'pgd' in results:
        eps = config['adversarial']['attacks']['pgd']['epsilon']
        idx = np.argmin(np.abs(np.array(results['pgd']['epsilon']) - eps))
        print(f"PGD Accuracy (ε={eps}): {results['pgd']['accuracy'][idx]:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model robustness")
    parser.add_argument("model_path", type=str,
                        help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--max-epsilon", type=float, default=0.1,
                        help="Maximum epsilon value to test")
    parser.add_argument("--num-epsilon-steps", type=int, default=11,
                        help="Number of epsilon values to test")
    parser.add_argument("--pgd-steps", type=int, default=40,
                        help="Number of PGD steps")
    parser.add_argument("--pgd-alpha-ratio", type=float, default=2.5,
                        help="Ratio of epsilon to alpha for PGD")
    parser.add_argument("--skip-fgsm", action="store_true",
                        help="Skip FGSM evaluation")
    parser.add_argument("--skip-pgd", action="store_true",
                        help="Skip PGD evaluation")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip generating plots")
    parser.add_argument("--compare-attacks", action="store_true",
                        help="Compare different attacks on sample batch")
    parser.add_argument("--test-encryption", action="store_true",
                        help="Test homomorphic encryption inference")
    
    args = parser.parse_args()
    main(args)