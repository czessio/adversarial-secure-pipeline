# Model quantization implementation for efficient inference

import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Union
import copy
from tqdm import tqdm


class QuantizedModel:
    """Handler for model quantization."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise quantization handler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.quant_config = config['quantization']
        self.quant_type = self.quant_config['type']
        self.backend = self.quant_config['backend']
        
        # Set quantization backend
        torch.backends.quantized.engine = self.backend
    
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """
        Prepare model for quantization.
        
        Args:
            model: Model to prepare
        
        Returns:
            Prepared model
        """
        # Make a copy to avoid modifying original
        model_copy = copy.deepcopy(model)
        
        # Fuse modules for better quantization
        model_copy = self._fuse_modules(model_copy)
        
        if self.quant_type == 'dynamic':
            # Dynamic quantization doesn't need preparation
            return model_copy
        elif self.quant_type == 'static':
            # Prepare for static quantization
            model_copy.qconfig = quantization.get_default_qconfig(self.backend)
            quantization.prepare(model_copy, inplace=True)
            return model_copy
        else:
            raise ValueError(f"Unknown quantization type: {self.quant_type}")
    
    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """
        Fuse conv-bn-relu modules for efficient quantization.
        
        Args:
            model: Model to fuse
        
        Returns:
            Fused model
        """
        # This is architecture-specific
        # Example for ResNet
        if hasattr(model, 'backbone') and 'resnet' in self.config['model']['architecture']:
            modules_to_fuse = []
            
            # Fuse conv-bn-relu in ResNet blocks
            for name, module in model.backbone.named_modules():
                if isinstance(module, nn.Conv2d):
                    # Check if followed by BatchNorm
                    parts = name.split('.')
                    if len(parts) > 0:
                        parent_name = '.'.join(parts[:-1])
                        conv_name = parts[-1]
                        
                        try:
                            parent = model.backbone
                            for part in parts[:-1]:
                                parent = getattr(parent, part)
                            
                            # Check for bn and relu
                            if hasattr(parent, f'bn{conv_name[-1]}'):
                                bn_name = f'bn{conv_name[-1]}'
                                if hasattr(parent, 'relu'):
                                    modules_to_fuse.append([
                                        f'{parent_name}.{conv_name}',
                                        f'{parent_name}.{bn_name}',
                                        f'{parent_name}.relu'
                                    ])
                                else:
                                    modules_to_fuse.append([
                                        f'{parent_name}.{conv_name}',
                                        f'{parent_name}.{bn_name}'
                                    ])
                        except:
                            continue
            
            if modules_to_fuse:
                model = quantization.fuse_modules(model, modules_to_fuse)
        
        return model
    
    def calibrate_model(
        self,
        model: nn.Module,
        calibration_loader: DataLoader,
        device: torch.device = torch.device('cpu')
    ) -> nn.Module:
        """
        Calibrate model for static quantization.
        
        Args:
            model: Prepared model
            calibration_loader: Data loader for calibration
            device: Device to use
        
        Returns:
            Calibrated model
        """
        if self.quant_type != 'static':
            return model
        
        model.eval()
        model.to(device)
        
        print("Calibrating model for static quantization...")
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(tqdm(calibration_loader)):
                if batch_idx >= self.quant_config['calibration_batches']:
                    break
                data = data.to(device)
                _ = model(data)
        
        return model
    
    def quantize_model(
        self,
        model: nn.Module,
        calibration_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device('cpu')
    ) -> nn.Module:
        """
        Quantize model.
        
        Args:
            model: Model to quantize
            calibration_loader: Data loader for calibration (static quantization)
            device: Device to use
        
        Returns:
            Quantized model
        """
        # Prepare model
        prepared_model = self.prepare_model(model)
        
        if self.quant_type == 'dynamic':
            # Apply dynamic quantization
            quantized_model = quantization.quantize_dynamic(
                prepared_model,
                qconfig_spec={nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
        elif self.quant_type == 'static':
            if calibration_loader is None:
                raise ValueError("Calibration loader required for static quantization")
            
            # Calibrate model
            prepared_model = self.calibrate_model(prepared_model, calibration_loader, device)
            
            # Convert to quantized model
            quantized_model = quantization.convert(prepared_model)
        else:
            raise ValueError(f"Unknown quantization type: {self.quant_type}")
        
        return quantized_model
    
    def evaluate_quantized_model(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        test_loader: DataLoader,
        device: torch.device = torch.device('cpu')
    ) -> Dict[str, Any]:
        """
        Evaluate quantized model and compare with original.
        
        Args:
            original_model: Original model
            quantized_model: Quantized model
            test_loader: Test data loader
            device: Device to use
        
        Returns:
            Evaluation results
        """
        results = {
            'original': self._evaluate_model(original_model, test_loader, device),
            'quantized': self._evaluate_model(quantized_model, test_loader, device)
        }
        
        # Calculate compression ratio
        original_size = self._get_model_size(original_model)
        quantized_size = self._get_model_size(quantized_model)
        results['compression_ratio'] = original_size / quantized_size
        
        # Calculate speedup
        original_time = self._measure_inference_time(original_model, test_loader, device)
        quantized_time = self._measure_inference_time(quantized_model, test_loader, device)
        results['speedup'] = original_time / quantized_time
        
        return results
    
    def _evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """Evaluate model accuracy."""
        model.eval()
        model.to(device)
        
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
        return {'accuracy': accuracy}
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def _measure_inference_time(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        num_batches: int = 100
    ) -> float:
        """Measure average inference time."""
        model.eval()
        model.to(device)
        
        # Warm up
        for i, (data, _) in enumerate(test_loader):
            if i >= 10:
                break
            data = data.to(device)
            with torch.no_grad():
                _ = model(data)
        
        # Measure time
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
        end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
        
        if device.type == 'cuda':
            start_time.record()
        else:
            import time
            start_time = time.time()
        
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                if i >= num_batches:
                    break
                data = data.to(device)
                _ = model(data)
        
        if device.type == 'cuda':
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
        else:
            elapsed_time = time.time() - start_time
        
        return elapsed_time


def quantize_robust_model(
    model: nn.Module,
    config: Dict[str, Any],
    calibration_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    device: torch.device = torch.device('cpu'),
    save_path: Optional[str] = None
) -> nn.Module:
    """
    Quantize a robust model.
    
    Args:
        model: Model to quantize
        config: Configuration dictionary
        calibration_loader: Data loader for calibration
        test_loader: Test data loader for evaluation
        device: Device to use
        save_path: Path to save quantized model
    
    Returns:
        Quantized model
    """
    quantizer = QuantizedModel(config)
    
    # Quantize model
    quantized_model = quantizer.quantize_model(model, calibration_loader, device)
    
    # Evaluate if test loader provided
    if test_loader is not None:
        print("\nEvaluating quantized model...")
        results = quantizer.evaluate_quantized_model(model, quantized_model, test_loader, device)
        
        print(f"\nOriginal Model Accuracy: {results['original']['accuracy']:.2f}%")
        print(f"Quantized Model Accuracy: {results['quantized']['accuracy']:.2f}%")
        print(f"Compression Ratio: {results['compression_ratio']:.2f}x")
        print(f"Speedup: {results['speedup']:.2f}x")
    
    # Save quantized model
    if save_path is not None:
        torch.save({
            'model_state_dict': quantized_model.state_dict(),
            'config': config,
            'quantization_type': config['quantization']['type']
        }, save_path)
        print(f"\nSaved quantized model to {save_path}")
    
    return quantized_model