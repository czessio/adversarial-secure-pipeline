# Utility functions for Microsoft SEAL homomorphic encryption

import torch
import numpy as np
import tenseal as ts
from typing import List, Tuple, Dict, Any, Optional
import json


class SEALContextManager:
    """Manager for SEAL encryption contexts."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise SEAL context manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config['privacy']['homomorphic_encryption']
        self.contexts = {}
        self.default_context = None
        self._create_default_context()
    
    def _create_default_context(self):
        """Create default encryption context."""
        self.default_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.config['poly_modulus_degree'],
            coeff_mod_bit_sizes=self.config['coeff_mod_bit_sizes']
        )
        self.default_context.global_scale = 2 ** self.config['scale_bits']
        self.default_context.generate_galois_keys()
        
        self.contexts['default'] = self.default_context
    
    def create_context(
        self,
        name: str,
        poly_modulus_degree: Optional[int] = None,
        coeff_mod_bit_sizes: Optional[List[int]] = None,
        scale_bits: Optional[int] = None
    ) -> ts.Context:
        """
        Create named encryption context.
        
        Args:
            name: Context name
            poly_modulus_degree: Polynomial modulus degree
            coeff_mod_bit_sizes: Coefficient modulus bit sizes
            scale_bits: Scale bits
        
        Returns:
            Encryption context
        """
        # Use provided parameters or defaults
        poly_modulus_degree = poly_modulus_degree or self.config['poly_modulus_degree']
        coeff_mod_bit_sizes = coeff_mod_bit_sizes or self.config['coeff_mod_bit_sizes']
        scale_bits = scale_bits or self.config['scale_bits']
        
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        context.global_scale = 2 ** scale_bits
        context.generate_galois_keys()
        
        self.contexts[name] = context
        return context
    
    def get_context(self, name: str = 'default') -> ts.Context:
        """Get encryption context by name."""
        return self.contexts.get(name, self.default_context)
    
    def save_context(self, name: str, path: str):
        """Save context to file."""
        context = self.get_context(name)
        with open(path, 'wb') as f:
            f.write(context.serialize())
    
    def load_context(self, path: str, name: str) -> ts.Context:
        """Load context from file."""
        with open(path, 'rb') as f:
            context_data = f.read()
        context = ts.context_from(context_data)
        self.contexts[name] = context
        return context


class EncryptionKeyManager:
    """Manager for encryption keys."""
    
    def __init__(self, context_manager: SEALContextManager):
        """
        Initialise key manager.
        
        Args:
            context_manager: SEAL context manager
        """
        self.context_manager = context_manager
        self.public_keys = {}
        self.secret_keys = {}
        self.galois_keys = {}
    
    def generate_keys(self, context_name: str = 'default'):
        """Generate encryption keys for context."""
        context = self.context_manager.get_context(context_name)
        
        # Keys are automatically generated with context in TenSEAL
        # Store references
        self.public_keys[context_name] = context
        self.secret_keys[context_name] = context
        self.galois_keys[context_name] = context
    
    def make_context_public(self, context_name: str = 'default'):
        """Make context public (remove secret key)."""
        context = self.context_manager.get_context(context_name)
        context.make_context_public()


class HomomorphicOperations:
    """Common homomorphic operations."""
    
    @staticmethod
    def encrypted_dot_product(
        vec1: ts.CKKSVector,
        vec2: ts.CKKSVector
    ) -> ts.CKKSVector:
        """Compute encrypted dot product."""
        return vec1.dot(vec2)
    
    @staticmethod
    def encrypted_matrix_multiply(
        matrix: List[ts.CKKSVector],
        vector: ts.CKKSVector
    ) -> ts.CKKSVector:
        """Compute encrypted matrix-vector multiplication."""
        result = []
        for row in matrix:
            result.append(row.dot(vector))
        return ts.CKKSVector.pack_vectors(result)
    
    @staticmethod
    def encrypted_convolution_1d(
        signal: ts.CKKSVector,
        kernel: ts.CKKSVector,
        stride: int = 1
    ) -> ts.CKKSVector:
        """Compute 1D convolution on encrypted data."""
        signal_len = len(signal)
        kernel_len = len(kernel)
        output_len = (signal_len - kernel_len) // stride + 1
        
        result = []
        for i in range(0, output_len * stride, stride):
            # Extract window
            window = signal[i:i + kernel_len]
            # Compute dot product with kernel
            conv_result = window.dot(kernel)
            result.append(conv_result)
        
        return ts.CKKSVector.pack_vectors(result)
    
    @staticmethod
    def encrypted_polynomial_activation(
        x: ts.CKKSVector,
        coefficients: List[float]
    ) -> ts.CKKSVector:
        """Apply polynomial activation function."""
        result = coefficients[0]
        x_power = x
        
        for i in range(1, len(coefficients)):
            result = result + coefficients[i] * x_power
            if i < len(coefficients) - 1:
                x_power = x_power * x
        
        return result


class EncryptedModelParameters:
    """Container for encrypted model parameters."""
    
    def __init__(self, context: ts.Context):
        """
        Initialise encrypted parameters container.
        
        Args:
            context: Encryption context
        """
        self.context = context
        self.weights = {}
        self.biases = {}
    
    def encrypt_linear_layer(
        self,
        layer_name: str,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ):
        """Encrypt linear layer parameters."""
        # Encrypt weight matrix row by row
        encrypted_weights = []
        for row in weight:
            encrypted_row = ts.ckks_vector(self.context, row.tolist())
            encrypted_weights.append(encrypted_row)
        self.weights[layer_name] = encrypted_weights
        
        # Encrypt bias if present
        if bias is not None:
            self.biases[layer_name] = ts.ckks_vector(self.context, bias.tolist())
    
    def encrypt_conv_layer(
        self,
        layer_name: str,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ):
        """Encrypt convolutional layer parameters."""
        # Flatten conv filters for encryption
        out_channels, in_channels, *kernel_size = weight.shape
        
        encrypted_filters = []
        for out_ch in range(out_channels):
            filter_data = weight[out_ch].flatten().tolist()
            encrypted_filter = ts.ckks_vector(self.context, filter_data)
            encrypted_filters.append(encrypted_filter)
        
        self.weights[layer_name] = {
            'filters': encrypted_filters,
            'shape': weight.shape
        }
        
        if bias is not None:
            self.biases[layer_name] = ts.ckks_vector(self.context, bias.tolist())
    
    def save(self, path: str):
        """Save encrypted parameters."""
        # Note: TenSEAL doesn't directly support serialising dictionaries of vectors
        # This is a simplified version - in practice, you'd need custom serialisation
        data = {
            'context': self.context.serialize(),
            'layer_names': list(self.weights.keys())
        }
        
        with open(path + '.json', 'w') as f:
            json.dump(data, f)
        
        # Save individual encrypted parameters
        for layer_name in self.weights:
            # Save weights
            if isinstance(self.weights[layer_name], list):
                for i, w in enumerate(self.weights[layer_name]):
                    w.save(f"{path}_{layer_name}_weight_{i}.seal")
            
            # Save biases
            if layer_name in self.biases:
                self.biases[layer_name].save(f"{path}_{layer_name}_bias.seal")


def benchmark_homomorphic_operations(context: ts.Context) -> Dict[str, float]:
    """
    Benchmark common homomorphic operations.
    
    Args:
        context: Encryption context
    
    Returns:
        Benchmark results
    """
    import time
    
    results = {}
    vector_size = 1024
    num_iterations = 10
    
    # Create test vectors
    vec1 = np.random.randn(vector_size).tolist()
    vec2 = np.random.randn(vector_size).tolist()
    
    # Encryption benchmark
    start = time.time()
    for _ in range(num_iterations):
        enc_vec1 = ts.ckks_vector(context, vec1)
    results['encryption_time'] = (time.time() - start) / num_iterations
    
    # Encrypt vectors for operations
    enc_vec1 = ts.ckks_vector(context, vec1)
    enc_vec2 = ts.ckks_vector(context, vec2)
    
    # Addition benchmark
    start = time.time()
    for _ in range(num_iterations):
        _ = enc_vec1 + enc_vec2
    results['addition_time'] = (time.time() - start) / num_iterations
    
    # Multiplication benchmark
    start = time.time()
    for _ in range(num_iterations):
        _ = enc_vec1 * enc_vec2
    results['multiplication_time'] = (time.time() - start) / num_iterations
    
    # Dot product benchmark
    start = time.time()
    for _ in range(num_iterations):
        _ = enc_vec1.dot(enc_vec2)
    results['dot_product_time'] = (time.time() - start) / num_iterations
    
    # Decryption benchmark
    start = time.time()
    for _ in range(num_iterations):
        _ = enc_vec1.decrypt()
    results['decryption_time'] = (time.time() - start) / num_iterations
    
    return results


def estimate_noise_budget(
    context: ts.Context,
    operations: List[str],
    initial_vector_size: int = 100
) -> Dict[str, Any]:
    """
    Estimate noise budget consumption for sequence of operations.
    
    Args:
        context: Encryption context
        operations: List of operations to perform
        initial_vector_size: Size of initial vector
    
    Returns:
        Noise budget analysis
    """
    # Create test vector
    test_vector = np.random.randn(initial_vector_size).tolist()
    encrypted = ts.ckks_vector(context, test_vector)
    
    noise_levels = []
    operation_count = 0
    
    for op in operations:
        try:
            if op == 'add':
                encrypted = encrypted + encrypted
            elif op == 'multiply':
                encrypted = encrypted * encrypted
            elif op == 'square':
                encrypted = encrypted.square()
            elif op == 'negate':
                encrypted = -encrypted
            
            operation_count += 1
            
            # Try to decrypt to check if noise budget exhausted
            decrypted = encrypted.decrypt()
            error = np.mean(np.abs(np.array(decrypted[:10]) - np.array(test_vector[:10])))
            noise_levels.append(error)
            
        except Exception as e:
            print(f"Noise budget exhausted after {operation_count} operations")
            break
    
    return {
        'max_operations': operation_count,
        'noise_progression': noise_levels,
        'final_error': noise_levels[-1] if noise_levels else float('inf')
    }