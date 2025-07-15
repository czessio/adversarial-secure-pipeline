# Homomorphic encryption for privacy-preserving inference using Microsoft SEAL

import torch
import torch.nn as nn
import numpy as np
import tenseal as ts
from typing import Dict, Any, List, Tuple, Optional
import time


class HomomorphicInference:
    """Privacy-preserving inference using homomorphic encryption."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise homomorphic encryption handler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.he_config = config['privacy']['homomorphic_encryption']
        self.context = None
        self._setup_context()
    
    def _setup_context(self):
        """Set up TenSEAL context for homomorphic encryption."""
        # Create context with specified parameters
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.he_config['poly_modulus_degree'],
            coeff_mod_bit_sizes=self.he_config['coeff_mod_bit_sizes']
        )
        
        # Set scale for CKKS
        self.context.global_scale = 2 ** self.he_config['scale_bits']
        
        # Generate galois keys for rotations (needed for convolutions)
        self.context.generate_galois_keys()
    
    def encrypt_tensor(self, tensor: torch.Tensor) -> ts.CKKSTensor:
        """
        Encrypt a tensor.
        
        Args:
            tensor: Tensor to encrypt
        
        Returns:
            Encrypted tensor
        """
        # Convert to numpy if needed
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()
        
        # Flatten tensor for encryption
        flat_tensor = tensor.flatten().tolist()
        
        # Encrypt using CKKS
        if self.he_config['batch_encode']:
            encrypted = ts.ckks_tensor(self.context, flat_tensor)
        else:
            encrypted = ts.ckks_vector(self.context, flat_tensor)
        
        # Store original shape for reconstruction
        encrypted.shape = tensor.shape
        
        return encrypted
    
    def decrypt_tensor(self, encrypted_tensor: ts.CKKSTensor) -> torch.Tensor:
        """
        Decrypt an encrypted tensor.
        
        Args:
            encrypted_tensor: Encrypted tensor
        
        Returns:
            Decrypted tensor
        """
        # Decrypt to list
        decrypted = encrypted_tensor.decrypt()
        
        # Reshape to original shape
        if hasattr(encrypted_tensor, 'shape'):
            decrypted = np.array(decrypted).reshape(encrypted_tensor.shape)
        else:
            decrypted = np.array(decrypted)
        
        return torch.from_numpy(decrypted).float()
    
    def encrypt_model_weights(self, model: nn.Module) -> Dict[str, ts.CKKSTensor]:
        """
        Encrypt model weights.
        
        Args:
            model: Model whose weights to encrypt
        
        Returns:
            Dictionary of encrypted weights
        """
        encrypted_weights = {}
        
        for name, param in model.named_parameters():
            encrypted_weights[name] = self.encrypt_tensor(param.data)
            print(f"Encrypted {name}: {param.shape}")
        
        return encrypted_weights
    
    def create_encrypted_linear_layer(
        self,
        weight: ts.CKKSTensor,
        bias: Optional[ts.CKKSTensor] = None
    ):
        """
        Create function for encrypted linear layer computation.
        
        Args:
            weight: Encrypted weight matrix
            bias: Encrypted bias vector (optional)
        
        Returns:
            Function that performs encrypted linear transformation
        """
        def encrypted_forward(x: ts.CKKSTensor) -> ts.CKKSTensor:
            # Matrix multiplication in encrypted domain
            output = x.mm(weight)
            
            # Add bias if provided
            if bias is not None:
                output = output + bias
            
            return output
        
        return encrypted_forward
    
    def approximate_activation(
        self,
        x: ts.CKKSTensor,
        activation: str = 'relu',
        degree: int = 3
    ) -> ts.CKKSTensor:
        """
        Approximate activation function using polynomial.
        
        Args:
            x: Encrypted input
            activation: Type of activation ('relu', 'sigmoid', 'tanh')
            degree: Degree of polynomial approximation
        
        Returns:
            Encrypted output after activation
        """
        if activation == 'relu':
            # Polynomial approximation of ReLU
            # Using x + 0.5*x^2 for x in [-1, 1]
            return x + 0.5 * x.square()
        
        elif activation == 'sigmoid':
            # Taylor series approximation of sigmoid
            # sigmoid(x) ≈ 0.5 + 0.25*x - 0.03125*x^3
            x_squared = x.square()
            x_cubed = x * x_squared
            return 0.5 + 0.25 * x - 0.03125 * x_cubed
        
        elif activation == 'tanh':
            # Taylor series approximation of tanh
            # tanh(x) ≈ x - x^3/3
            x_cubed = x.square() * x
            return x - x_cubed / 3
        
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def encrypted_inference_simple(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        max_layers: int = 3
    ) -> torch.Tensor:
        """
        Perform encrypted inference on a simple model.
        
        Args:
            model: Model to use (must be simple architecture)
            input_data: Input data
            max_layers: Maximum number of layers to process
        
        Returns:
            Decrypted output
        """
        # Encrypt input
        x = self.encrypt_tensor(input_data)
        
        # Process through model layers
        layer_count = 0
        for name, module in model.named_children():
            if layer_count >= max_layers:
                break
            
            if isinstance(module, nn.Linear):
                # Encrypt weights
                weight = self.encrypt_tensor(module.weight.data.T)  # Transpose for mm
                bias = self.encrypt_tensor(module.bias.data) if module.bias is not None else None
                
                # Linear transformation
                x = x.mm(weight)
                if bias is not None:
                    x = x + bias
                
                layer_count += 1
                
            elif isinstance(module, nn.ReLU):
                # Approximate ReLU
                x = self.approximate_activation(x, 'relu')
            
            elif isinstance(module, nn.Sequential):
                # Process sequential blocks
                for submodule in module:
                    if isinstance(submodule, nn.Linear) and layer_count < max_layers:
                        weight = self.encrypt_tensor(submodule.weight.data.T)
                        bias = self.encrypt_tensor(submodule.bias.data) if submodule.bias is not None else None
                        x = x.mm(weight)
                        if bias is not None:
                            x = x + bias
                        layer_count += 1
                    elif isinstance(submodule, nn.ReLU):
                        x = self.approximate_activation(x, 'relu')
        
        # Decrypt result
        return self.decrypt_tensor(x)
    
    def benchmark_encrypted_inference(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark encrypted inference performance.
        
        Args:
            model: Model to benchmark
            input_shape: Shape of input data
            num_runs: Number of runs for averaging
        
        Returns:
            Benchmark results
        """
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Time regular inference
        model.eval()
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(dummy_input)
        regular_time = (time.time() - start_time) / num_runs
        
        # Time encrypted inference
        start_time = time.time()
        for _ in range(num_runs):
            _ = self.encrypted_inference_simple(model, dummy_input)
        encrypted_time = (time.time() - start_time) / num_runs
        
        return {
            'regular_inference_time': regular_time,
            'encrypted_inference_time': encrypted_time,
            'overhead_factor': encrypted_time / regular_time
        }


class SecureModelServer:
    """Server for secure model inference."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """
        Initialise secure model server.
        
        Args:
            model: Model to serve
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.he_handler = HomomorphicInference(config)
        
        # Pre-encrypt model weights for efficiency
        self.encrypted_weights = self.he_handler.encrypt_model_weights(model)
    
    def process_encrypted_request(
        self,
        encrypted_input: ts.CKKSTensor
    ) -> ts.CKKSTensor:
        """
        Process encrypted inference request.
        
        Args:
            encrypted_input: Encrypted input from client
        
        Returns:
            Encrypted output
        """
        # Perform inference in encrypted domain
        x = encrypted_input
        
        # Simple forward pass (limited to linear layers for now)
        for name, module in self.model.named_children():
            if isinstance(module, nn.Linear):
                # Use pre-encrypted weights
                weight_name = f"{name}.weight"
                bias_name = f"{name}.bias"
                
                if weight_name in self.encrypted_weights:
                    x = x.mm(self.encrypted_weights[weight_name])
                    
                    if bias_name in self.encrypted_weights:
                        x = x + self.encrypted_weights[bias_name]
            
            elif isinstance(module, nn.ReLU):
                x = self.he_handler.approximate_activation(x, 'relu')
        
        return x


class SecureModelClient:
    """Client for secure model inference."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise secure model client.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.he_handler = HomomorphicInference(config)
    
    def prepare_encrypted_input(self, input_data: torch.Tensor) -> ts.CKKSTensor:
        """
        Prepare encrypted input for server.
        
        Args:
            input_data: Input data to encrypt
        
        Returns:
            Encrypted input
        """
        return self.he_handler.encrypt_tensor(input_data)
    
    def decrypt_response(self, encrypted_output: ts.CKKSTensor) -> torch.Tensor:
        """
        Decrypt server response.
        
        Args:
            encrypted_output: Encrypted output from server
        
        Returns:
            Decrypted output
        """
        return self.he_handler.decrypt_tensor(encrypted_output)


def demonstrate_private_inference(
    model: nn.Module,
    test_input: torch.Tensor,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Demonstrate privacy-preserving inference.
    
    Args:
        model: Model to use
        test_input: Test input
        config: Configuration dictionary
    
    Returns:
        Results dictionary
    """
    print("Setting up secure inference...")
    
    # Create client and server
    client = SecureModelClient(config)
    server = SecureModelServer(model, config)
    
    # Client encrypts input
    print("Client encrypting input...")
    encrypted_input = client.prepare_encrypted_input(test_input)
    
    # Server processes encrypted input
    print("Server processing encrypted data...")
    encrypted_output = server.process_encrypted_request(encrypted_input)
    
    # Client decrypts result
    print("Client decrypting result...")
    decrypted_output = client.decrypt_response(encrypted_output)
    
    # Compare with regular inference
    model.eval()
    with torch.no_grad():
        regular_output = model(test_input)
    
    # Calculate error
    error = torch.mean(torch.abs(decrypted_output - regular_output))
    
    return {
        'encrypted_output': decrypted_output,
        'regular_output': regular_output,
        'absolute_error': error.item(),
        'success': error.item() < 0.1  # Threshold for acceptable error
    }