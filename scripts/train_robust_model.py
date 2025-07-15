# Script to train adversarially robust model

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from pathlib import Path

from src.utils.data_loader import load_config, create_data_loaders
from src.models.base_classifier import create_model
from src.defences.adversarial_training import create_robust_model
from src.models.quantized_model import quantize_robust_model
from src.preprocessing.lighting_correction import create_lighting_corrected_transform
from src.utils.data_loader import get_data_transforms


def main(args):
    """Main training function."""
    # Load configuration
    config = load_config(args.config)
    
    # Update config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.attack_ratio:
        config['adversarial']['training']['attack_ratio'] = args.attack_ratio
    
    # Set device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders with lighting correction if enabled
    print("\nPreparing data loaders...")
    if args.use_lighting_correction:
        base_train_transform = get_data_transforms(config, train=True)
        base_val_transform = get_data_transforms(config, train=False)
        
        train_transform = create_lighting_corrected_transform(config, base_train_transform)
        val_transform = create_lighting_corrected_transform(config, base_val_transform)
        
        train_loader, val_loader, test_loader = create_data_loaders(
            config,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=val_transform
        )
    else:
        train_loader, val_loader, test_loader = create_data_loaders(config)
    
    print(f"Dataset: {config['data']['dataset_name']}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"\nCreating model: {config['model']['architecture']}...")
    model = create_model(config, ensemble=args.use_ensemble)
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Create save directory
    save_dir = Path(args.save_dir) / config['data']['dataset_name'] / config['model']['architecture']
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Train robust model
    if args.skip_training and args.checkpoint:
        print(f"\nLoading model from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        robust_model = model
    else:
        print("\nStarting adversarial training...")
        print(f"Attack ratio: {config['adversarial']['training']['attack_ratio']}")
        print(f"Max epsilon: {config['adversarial']['training']['max_epsilon']}")
        print(f"Epsilon schedule: {config['adversarial']['training']['epsilon_schedule']}")
        
        robust_model = create_robust_model(
            model,
            config,
            train_loader,
            val_loader,
            device,
            save_path=save_dir / "robust_model.pth",
            use_free_training=args.use_free_training
        )
    
    # Quantize model if requested
    if args.quantize:
        print("\nQuantizing model...")
        quantized_save_path = save_dir / "quantized_model.pth"
        
        # Use a subset of training data for calibration
        calibration_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers']
        )
        
        quantized_model = quantize_robust_model(
            robust_model,
            config,
            calibration_loader=calibration_loader if config['quantization']['type'] == 'static' else None,
            test_loader=test_loader,
            device=device,
            save_path=quantized_save_path
        )
    
    print("\nTraining complete!")
    print(f"Models saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train adversarially robust model")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--save-dir", type=str, default="models/checkpoints",
                        help="Directory to save trained models")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (overrides config)")
    parser.add_argument("--attack-ratio", type=float, default=None,
                        help="Ratio of adversarial examples in training (overrides config)")
    parser.add_argument("--use-lighting-correction", action="store_true",
                        help="Apply lighting correction preprocessing")
    parser.add_argument("--use-ensemble", action="store_true",
                        help="Use ensemble of models")
    parser.add_argument("--use-free-training", action="store_true",
                        help="Use free adversarial training")
    parser.add_argument("--quantize", action="store_true",
                        help="Quantize model after training")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training and load from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    
    args = parser.parse_args()
    main(args)