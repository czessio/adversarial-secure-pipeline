# Adversarially Robust Image Classification Pipeline

A comprehensive end-to-end pipeline for developing adversarially robust image classification models with advanced defence mechanisms, bias analysis, and privacy-preserving inference capabilities.

## Features

### 1. **Lighting Correction**
- **Retinex Algorithm**: Single-scale Retinex for illumination correction
- **Adaptive Histogram Equalisation (CLAHE)**: Contrast enhancement with local adaptation
- Configurable preprocessing pipeline with multiple correction methods

### 2. **Adversarial Attacks & Defences**
- **Attack Implementations**:
  - Fast Gradient Sign Method (FGSM)
  - Projected Gradient Descent (PGD)
  - Iterative FGSM (I-FGSM)
  - Multi-targeted PGD
- **Defence Mechanisms**:
  - Adversarial training with configurable epsilon schedules
  - Free adversarial training for efficiency
  - Input transformations (JPEG compression, bit depth reduction, Gaussian smoothing)
  - Defensive distillation
  - Ensemble defences

### 3. **Dataset Bias Analysis**
- Class distribution analysis across training/test sets
- Per-class performance metrics
- Robustness bias evaluation
- Lighting sensitivity analysis
- Comprehensive bias pattern identification

### 4. **Model Quantization**
- Dynamic quantization for inference optimisation
- Static quantization with calibration
- PyTorch quantization API integration
- Compression ratio and speedup analysis

### 5. **Privacy-Preserving Inference**
- Homomorphic encryption using Microsoft SEAL (via TenSEAL)
- Encrypted model inference
- Client-server architecture for secure computation
- Polynomial approximations for encrypted activations

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional but recommended)
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/adversarial-robust-pipeline.git
cd adversarial-robust-pipeline
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install Package
```bash
pip install -e .
```

### Step 5: Download Datasets
The pipeline will automatically download CIFAR-10/CIFAR-100 datasets on first run.

## Project Structure

```
adversarial_robust_pipeline/
├── config/
│   └── config.yaml              # Main configuration file
├── src/
│   ├── preprocessing/           # Data preprocessing modules
│   ├── attacks/                 # Adversarial attack implementations
│   ├── defences/                # Defence mechanisms
│   ├── models/                  # Model architectures
│   ├── privacy/                 # Privacy-preserving techniques
│   ├── analysis/                # Bias and robustness analysis
│   └── utils/                   # Utility functions
├── scripts/
│   ├── train_robust_model.py    # Training script
│   ├── evaluate_robustness.py   # Evaluation script
│   ├── analyse_bias.py          # Bias analysis script
│   └── run_pipeline.py          # Complete pipeline runner
├── data/                        # Data directory (auto-created)
├── models/                      # Saved models directory
├── results/                     # Results and visualisations
└── experiments/                 # Experiment tracking
```

## Quick Start

### 1. Run Complete Pipeline
```bash
python scripts/run_pipeline.py --full-pipeline --experiment-name my_experiment
```

### 2. Train Robust Model
```bash
python scripts/train_robust_model.py \
    --config config/config.yaml \
    --epochs 100 \
    --use-lighting-correction \
    --quantize
```

### 3. Evaluate Robustness
```bash
python scripts/evaluate_robustness.py \
    models/checkpoints/robust_model.pth \
    --compare-attacks \
    --test-encryption
```

### 4. Analyse Bias
```bash
python scripts/analyse_bias.py \
    --model-path models/checkpoints/robust_model.pth \
    --analyse-robustness \
    --analyse-lighting
```

## Configuration

The pipeline is configured via `config/config.yaml`. Key settings include:

```yaml
# Data settings
data:
  dataset_name: "CIFAR10"  # CIFAR10 or CIFAR100
  batch_size: 128
  validation_split: 0.2

# Model settings
model:
  architecture: "resnet18"  # resnet18, resnet34, resnet50, vgg16
  dropout_rate: 0.5

# Adversarial settings
adversarial:
  attacks:
    fgsm:
      epsilon: 0.03
    pgd:
      epsilon: 0.03
      alpha: 0.01
      num_steps: 40
  training:
    enable: true
    attack_ratio: 0.5
    epsilon_schedule: "linear"

# Quantization settings
quantization:
  enable: true
  type: "dynamic"  # dynamic or static

# Privacy settings
privacy:
  homomorphic_encryption:
    enable: true
    poly_modulus_degree: 8192
```

## Advanced Usage

### Custom Attack Implementation
```python
from src.attacks.fgsm import FGSMAttack

# Create custom attack
attack = FGSMAttack(epsilon=0.05, targeted=True)

# Generate adversarial examples
adv_examples = attack.generate(model, images, labels, target_labels)
```

### Bias Analysis
```python
from src.analysis.bias_analysis import BiasAnalyser

# Create analyser
analyser = BiasAnalyser(config, class_names)

# Generate comprehensive report
bias_report = analyser.generate_bias_report(model, train_loader, test_loader)

# Identify bias patterns
patterns = analyser.identify_bias_patterns(bias_report)
```

### Privacy-Preserving Inference
```python
from src.privacy.homomorphic_encryption import SecureModelClient, SecureModelServer

# Client side
client = SecureModelClient(config)
encrypted_input = client.prepare_encrypted_input(image)

# Server side
server = SecureModelServer(model, config)
encrypted_output = server.process_encrypted_request(encrypted_input)

# Client decrypts result
result = client.decrypt_response(encrypted_output)
```

## Experimental Results

The pipeline provides comprehensive evaluation metrics:

- **Clean Accuracy**: Performance on unperturbed images
- **Adversarial Robustness**: Accuracy under FGSM/PGD attacks
- **Efficiency Metrics**: Inference time, model size, FLOPs
- **Bias Analysis**: Class-wise performance disparities
- **Privacy Overhead**: Homomorphic encryption computational cost

## Visualisations

The pipeline generates various visualisations:
- Training curves (loss, accuracy, robustness)
- Confusion matrices
- Per-class performance charts
- Robustness curves across epsilon values
- Bias analysis dashboards
- Feature space visualisations (t-SNE/PCA)

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in config.yaml
- Enable mixed precision training
- Use gradient accumulation

### Homomorphic Encryption Issues
- Ensure TenSEAL is properly installed
- Check SEAL library compatibility
- Reduce polynomial modulus degree for testing

### Slow Training
- Enable free adversarial training
- Reduce PGD steps
- Use dynamic quantization instead of static

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{adversarial_robust_pipeline,
  title = {Adversarially Robust Image Classification Pipeline},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/adversarial-robust-pipeline}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- Microsoft Research for the SEAL homomorphic encryption library
- The adversarial robustness research community

## Contact

For questions or support, please open an issue on GitHub or contact: ...