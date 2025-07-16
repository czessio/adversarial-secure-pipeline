#!/usr/bin/env python
"""
Test script to verify that all fixes are working correctly.
Run this before running the full pipeline to ensure everything is set up properly.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import load_config, create_data_loaders
from src.models.base_classifier import create_model
from src.attacks.fgsm import FGSMAttack
from src.preprocessing.lighting_correction import RetinexCorrection


def test_device_availability():
    """Test CUDA availability and device setup."""
    print("="*50)
    print("Testing Device Availability")
    print("="*50)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Test tensor creation on device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        test_tensor = torch.randn(10, 10, device=device)
        print(f"Successfully created tensor on {device}")
    except Exception as e:
        print(f"Error creating tensor on {device}: {e}")
        return False
    
    return True


def test_model_device_placement():
    """Test model device placement."""
    print("\n" + "="*50)
    print("Testing Model Device Placement")
    print("="*50)
    
    config = load_config()
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Check if model is on correct device
    first_param_device = next(model.parameters()).device
    print(f"Model device: {first_param_device}")
    print(f"Expected device: {device}")
    
    if str(first_param_device) != str(device):
        print("ERROR: Model not on correct device!")
        return False
    
    # Test forward pass
    try:
        dummy_input = torch.randn(1, 3, 32, 32, device=device)
        output = model(dummy_input)
        print(f"Forward pass successful. Output shape: {output.shape}")
    except Exception as e:
        print(f"Error in forward pass: {e}")
        return False
    
    return True


def test_loss_calculation():
    """Test loss calculation for numerical stability."""
    print("\n" + "="*50)
    print("Testing Loss Calculation")
    print("="*50)
    
    # Create dummy data
    outputs = torch.randn(32, 10)
    targets = torch.randint(0, 10, (32,))
    
    # Calculate loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
    
    print(f"Loss value: {loss.item():.4f}")
    
    # Check for NaN or Inf
    if torch.isnan(loss) or torch.isinf(loss):
        print("ERROR: Loss is NaN or Inf!")
        return False
    
    # Check reasonable range
    if loss.item() > 10:
        print("WARNING: Loss seems unusually high")
    
    return True


def test_adversarial_attack():
    """Test adversarial attack implementation."""
    print("\n" + "="*50)
    print("Testing Adversarial Attack")
    print("="*50)
    
    config = load_config()
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    
    # Create model and attack
    model = create_model(config).to(device)
    model.eval()
    attack = FGSMAttack(epsilon=0.03)
    
    # Create dummy data
    data = torch.randn(4, 3, 32, 32, device=device)
    target = torch.randint(0, 10, (4,), device=device)
    
    try:
        # Generate adversarial examples
        adv_data = attack.generate(model, data, target)
        
        # Check perturbation
        perturbation = (adv_data - data).abs().max().item()
        print(f"Max perturbation: {perturbation:.4f}")
        print(f"Expected max: {attack.epsilon}")
        
        if perturbation > attack.epsilon * 1.01:  # Small tolerance
            print("ERROR: Perturbation exceeds epsilon!")
            return False
            
    except Exception as e:
        print(f"Error in adversarial attack: {e}")
        return False
    
    return True


def test_lighting_correction():
    """Test lighting correction numerical stability."""
    print("\n" + "="*50)
    print("Testing Lighting Correction")
    print("="*50)
    
    # Create test image
    image = np.random.rand(32, 32, 3).astype(np.float32)
    
    # Apply Retinex correction
    retinex = RetinexCorrection(sigma=50)
    
    try:
        corrected = retinex(image)
        
        # Check for NaN or Inf
        if np.any(np.isnan(corrected)) or np.any(np.isinf(corrected)):
            print("ERROR: Lighting correction produced NaN or Inf!")
            return False
        
        # Check range
        print(f"Corrected image range: [{corrected.min():.4f}, {corrected.max():.4f}]")
        
        if corrected.min() < -0.1 or corrected.max() > 1.1:
            print("WARNING: Corrected image outside expected range [0, 1]")
            
    except Exception as e:
        print(f"Error in lighting correction: {e}")
        return False
    
    return True


def test_data_loading():
    """Test data loading pipeline."""
    print("\n" + "="*50)
    print("Testing Data Loading")
    print("="*50)
    
    config = load_config()
    
    try:
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(config)
        
        # Test loading a batch
        data, target = next(iter(train_loader))
        print(f"Batch shape: {data.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Data range: [{data.min():.4f}, {data.max():.4f}]")
        
        # Check data is normalised
        if data.min() < -3 or data.max() > 3:
            print("WARNING: Data might not be properly normalised")
            
    except Exception as e:
        print(f"Error in data loading: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("Running Adversarial Robustness Pipeline Tests")
    print("="*60)
    
    tests = [
        ("Device Availability", test_device_availability),
        ("Model Device Placement", test_model_device_placement),
        ("Loss Calculation", test_loss_calculation),
        ("Adversarial Attack", test_adversarial_attack),
        ("Lighting Correction", test_lighting_correction),
        ("Data Loading", test_data_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\nERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\nAll tests passed! The pipeline should work correctly.")
    else:
        print("\nSome tests failed. Please fix the issues before running the pipeline.")
    
    return all_passed


if __name__ == "__main__":
    main()