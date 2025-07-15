# Main script to run the complete adversarial robustness pipeline

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from pathlib import Path
import time
import json
from datetime import datetime

from src.utils.data_loader import load_config, create_data_loaders, get_class_names
from src.preprocessing.lighting_correction import create_lighting_corrected_transform
from src.preprocessing.data_augmentation import AugmentationWrapper
from src.models.base_classifier import create_model
from src.models.robust_classifier import create_robust_model_architecture
from src.defences.adversarial_training import AdversarialTrainer
from src.models.quantized_model import quantize_robust_model
from src.attacks.fgsm import FGSMAttack
from src.attacks.pgd import PGDAttack
from src.analysis.bias_analysis import BiasAnalyser
from src.analysis.visualisation import create_model_report, setup_plotting_style
from src.privacy.homomorphic_encryption import demonstrate_private_inference
from src.utils.metrics import RobustnessEvaluator, calculate_efficiency_metrics
from src.utils.training_utils import train_model


class AdversarialRobustnessPipeline:
    """Complete pipeline for adversarially robust image classification."""
    
    def __init__(self, config_path: str, experiment_name: str):
        """
        Initialise pipeline.
        
        Args:
            config_path: Path to configuration file
            experiment_name: Name for this experiment
        """
        self.config = load_config(config_path)
        self.experiment_name = experiment_name
        self.device = torch.device(
            self.config['hardware']['device'] if torch.cuda.is_available() else 'cpu'
        )
        
        # Create experiment directory
        self.experiment_dir = Path('experiments') / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(self.experiment_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Pipeline components
        self.model = None
        self.robust_model = None
        self.quantized_model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        print(f"Pipeline initialised: {experiment_name}")
        print(f"Device: {self.device}")
        print(f"Experiment directory: {self.experiment_dir}")
    
    def prepare_data(self, use_lighting_correction: bool = True):
        """Prepare data loaders with preprocessing."""
        print("\n" + "="*50)
        print("DATA PREPARATION")
        print("="*50)
        
        # Create data loaders
        if use_lighting_correction:
            print("Applying lighting correction...")
            from src.utils.data_loader import get_data_transforms
            
            base_train_transform = get_data_transforms(self.config, train=True)
            base_val_transform = get_data_transforms(self.config, train=False)
            
            train_transform = create_lighting_corrected_transform(
                self.config, base_train_transform
            )
            val_transform = create_lighting_corrected_transform(
                self.config, base_val_transform
            )
            
            self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
                self.config,
                train_transform=train_transform,
                val_transform=val_transform,
                test_transform=val_transform
            )
        else:
            self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
                self.config
            )
        
        print(f"Dataset: {self.config['data']['dataset_name']}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
    
    def train_base_model(self):
        """Train base model without adversarial training."""
        print("\n" + "="*50)
        print("BASE MODEL TRAINING")
        print("="*50)
        
        # Create model
        self.model = create_model(self.config)
        print(f"Architecture: {self.config['model']['architecture']}")
        print(f"Parameters: {self.model.get_num_parameters():,}")
        
        # Create augmentation wrapper
        augmentation = AugmentationWrapper(
            self.config,
            use_mixup=True,
            use_cutmix=True,
            use_autoaugment=True
        )
        
        # Train model
        self.model = train_model(
            self.model,
            self.train_loader,
            self.val_loader,
            self.config,
            self.device,
            experiment_name=f"{self.experiment_name}_base",
            augmentation_fn=augmentation
        )
        
        # Save model
        model_path = self.experiment_dir / 'base_model.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, model_path)
        print(f"Base model saved to: {model_path}")
    
    def train_robust_model(self, use_pretrained: bool = True):
        """Train adversarially robust model."""
        print("\n" + "="*50)
        print("ADVERSARIAL TRAINING")
        print("="*50)
        
        # Create or load model
        if use_pretrained and self.model is not None:
            print("Using pretrained base model...")
            self.robust_model = self.model
        else:
            print("Creating new robust model architecture...")
            self.robust_model = create_robust_model_architecture(
                self.config, model_type='adversarial'
            )
        
        # Adversarial trainer
        trainer = AdversarialTrainer(
            self.robust_model,
            self.config,
            self.device
        )
        
        print(f"Adversarial training configuration:")
        print(f"  - Attack ratio: {self.config['adversarial']['training']['attack_ratio']}")
        print(f"  - Max epsilon: {self.config['adversarial']['training']['max_epsilon']}")
        print(f"  - Epsilon schedule: {self.config['adversarial']['training']['epsilon_schedule']}")
        
        # Train
        trainer.train(
            self.train_loader,
            self.val_loader,
            save_path=self.experiment_dir / 'robust_model.pth'
        )
        
        self.robust_model = trainer.model
    
    def quantize_model(self):
        """Quantize the robust model."""
        print("\n" + "="*50)
        print("MODEL QUANTIZATION")
        print("="*50)
        
        if self.robust_model is None:
            print("No robust model available for quantization!")
            return
        
        # Prepare calibration loader
        calibration_loader = torch.utils.data.DataLoader(
            self.train_loader.dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers']
        )
        
        # Quantize
        self.quantized_model = quantize_robust_model(
            self.robust_model,
            self.config,
            calibration_loader=calibration_loader if self.config['quantization']['type'] == 'static' else None,
            test_loader=self.test_loader,
            device=self.device,
            save_path=self.experiment_dir / 'quantized_model.pth'
        )
    
    def evaluate_robustness(self):
        """Comprehensive robustness evaluation."""
        print("\n" + "="*50)
        print("ROBUSTNESS EVALUATION")
        print("="*50)
        
        models_to_evaluate = []
        
        if self.model is not None:
            models_to_evaluate.append(('Base Model', self.model))
        if self.robust_model is not None:
            models_to_evaluate.append(('Robust Model', self.robust_model))
        if self.quantized_model is not None:
            models_to_evaluate.append(('Quantized Model', self.quantized_model))
        
        # Create attacks
        attacks = [
            FGSMAttack(epsilon=self.config['adversarial']['attacks']['fgsm']['epsilon']),
            PGDAttack(
                epsilon=self.config['adversarial']['attacks']['pgd']['epsilon'],
                alpha=self.config['adversarial']['attacks']['pgd']['alpha'],
                num_steps=self.config['adversarial']['attacks']['pgd']['num_steps']
            )
        ]
        
        # Evaluate each model
        results = {}
        for model_name, model in models_to_evaluate:
            print(f"\nEvaluating {model_name}...")
            
            evaluator = RobustnessEvaluator(model, self.config)
            eval_results = evaluator.evaluate_comprehensive(self.test_loader, attacks)
            
            # Store results
            results[model_name] = eval_results
            
            # Save results
            eval_results.to_csv(
                self.experiment_dir / f'{model_name.lower().replace(" ", "_")}_evaluation.csv',
                index=False
            )
            
            # Print summary
            clean_acc = eval_results[eval_results['attack'] == 'None']['accuracy'].values[0]
            fgsm_acc = eval_results[
                (eval_results['attack'] == 'FGSMAttack') & 
                (eval_results['epsilon'] == self.config['adversarial']['attacks']['fgsm']['epsilon'])
            ]['accuracy'].values[0]
            pgd_acc = eval_results[
                (eval_results['attack'] == 'PGDAttack') & 
                (eval_results['epsilon'] == self.config['adversarial']['attacks']['pgd']['epsilon'])
            ]['accuracy'].values[0]
            
            print(f"  Clean accuracy: {clean_acc:.2f}%")
            print(f"  FGSM robustness: {fgsm_acc:.2f}%")
            print(f"  PGD robustness: {pgd_acc:.2f}%")
        
        return results
    
    def analyse_bias(self):
        """Perform bias analysis."""
        print("\n" + "="*50)
        print("BIAS ANALYSIS")
        print("="*50)
        
        if self.robust_model is None:
            print("No model available for bias analysis!")
            return None
        
        # Get class names
        class_names = get_class_names(self.config['data']['dataset_name'])
        
        # Create analyser
        analyser = BiasAnalyser(self.config, class_names)
        
        # Generate bias report
        bias_report = analyser.generate_bias_report(
            self.robust_model,
            self.train_loader,
            self.test_loader
        )
        
        # Save report
        bias_report.to_csv(self.experiment_dir / 'bias_analysis.csv', index=False)
        
        # Identify patterns
        patterns = analyser.identify_bias_patterns(bias_report)
        
        print("\nIdentified bias patterns:")
        for pattern_name, classes in patterns.items():
            if classes:
                print(f"  - {pattern_name.replace('_', ' ').title()}: {', '.join(classes)}")
        
        return bias_report
    
    def test_privacy_preservation(self):
        """Test homomorphic encryption for privacy-preserving inference."""
        print("\n" + "="*50)
        print("PRIVACY-PRESERVING INFERENCE TEST")
        print("="*50)
        
        if not self.config['privacy']['homomorphic_encryption']['enable']:
            print("Homomorphic encryption is disabled in configuration.")
            return
        
        if self.robust_model is None:
            print("No model available for privacy testing!")
            return
        
        # Get sample input
        sample_data, _ = next(iter(self.test_loader))
        sample_input = sample_data[0:1]  # Single sample
        
        print("Testing homomorphic encryption inference...")
        try:
            results = demonstrate_private_inference(
                self.robust_model,
                sample_input,
                self.config
            )
            
            print(f"Success: {results['success']}")
            print(f"Absolute error: {results['absolute_error']:.6f}")
            
            # Save results
            with open(self.experiment_dir / 'privacy_test_results.json', 'w') as f:
                json.dump({
                    'success': results['success'],
                    'absolute_error': results['absolute_error']
                }, f, indent=2)
        
        except Exception as e:
            print(f"Privacy test failed: {str(e)}")
            print("Note: Homomorphic encryption support may require additional setup.")
    
    def measure_efficiency(self):
        """Measure model efficiency metrics."""
        print("\n" + "="*50)
        print("EFFICIENCY ANALYSIS")
        print("="*50)
        
        models_to_analyse = []
        
        if self.model is not None:
            models_to_analyse.append(('Base Model', self.model))
        if self.robust_model is not None:
            models_to_analyse.append(('Robust Model', self.robust_model))
        if self.quantized_model is not None:
            models_to_analyse.append(('Quantized Model', self.quantized_model))
        
        # Input shape
        sample_data, _ = next(iter(self.test_loader))
        input_shape = sample_data[0].shape
        
        efficiency_results = {}
        
        for model_name, model in models_to_analyse:
            print(f"\nAnalysing {model_name}...")
            
            metrics = calculate_efficiency_metrics(
                model,
                (1, *input_shape),
                self.device
            )
            
            efficiency_results[model_name] = metrics
            
            print(f"  Parameters: {metrics['total_parameters']:,}")
            print(f"  Model size: {metrics['model_size_mb']:.2f} MB")
            print(f"  Inference time: {metrics['inference_time_ms']:.2f} ms")
            print(f"  Throughput: {metrics['throughput_fps']:.2f} FPS")
            print(f"  FLOPs: {metrics['flops_formatted']}")
        
        # Save results
        with open(self.experiment_dir / 'efficiency_metrics.json', 'w') as f:
            json.dump(efficiency_results, f, indent=2)
        
        return efficiency_results
    
    def generate_report(self, evaluation_results, bias_report):
        """Generate comprehensive report."""
        print("\n" + "="*50)
        print("GENERATING REPORT")
        print("="*50)
        
        # Prepare evaluation summary
        eval_summary = {}
        if 'Robust Model' in evaluation_results:
            results_df = evaluation_results['Robust Model']
            eval_summary['clean_accuracy'] = results_df[
                results_df['attack'] == 'None'
            ]['accuracy'].values[0]
            eval_summary['fgsm_accuracy'] = results_df[
                (results_df['attack'] == 'FGSMAttack') & 
                (results_df['epsilon'] == self.config['adversarial']['attacks']['fgsm']['epsilon'])
            ]['accuracy'].values[0]
            eval_summary['pgd_accuracy'] = results_df[
                (results_df['attack'] == 'PGDAttack') & 
                (results_df['epsilon'] == self.config['adversarial']['attacks']['pgd']['epsilon'])
            ]['accuracy'].values[0]
        
        # Create report
        create_model_report(
            self.config,
            eval_summary,
            bias_report,
            str(self.experiment_dir)
        )
        
        print(f"Report generated in: {self.experiment_dir}")
    
    def run_full_pipeline(self):
        """Run the complete pipeline."""
        start_time = time.time()
        
        print("\n" + "="*60)
        print("ADVERSARIAL ROBUSTNESS PIPELINE")
        print("="*60)
        print(f"Experiment: {self.experiment_name}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. Data preparation
        self.prepare_data(use_lighting_correction=True)
        
        # 2. Train base model
        if not self.config.get('skip_base_training', False):
            self.train_base_model()
        
        # 3. Adversarial training
        self.train_robust_model(use_pretrained=True)
        
        # 4. Model quantization
        if self.config['quantization']['enable']:
            self.quantize_model()
        
        # 5. Robustness evaluation
        evaluation_results = self.evaluate_robustness()
        
        # 6. Bias analysis
        bias_report = self.analyse_bias()
        
        # 7. Privacy preservation test
        self.test_privacy_preservation()
        
        # 8. Efficiency analysis
        self.measure_efficiency()
        
        # 9. Generate report
        self.generate_report(evaluation_results, bias_report)
        
        # Pipeline summary
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Results saved to: {self.experiment_dir}")
        
        # Save pipeline summary
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_time_minutes': total_time / 60,
            'device': str(self.device),
            'dataset': self.config['data']['dataset_name'],
            'architecture': self.config['model']['architecture'],
            'adversarial_training': self.config['adversarial']['training']['enable'],
            'quantization': self.config['quantization']['enable']
        }
        
        with open(self.experiment_dir / 'pipeline_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


def main(args):
    """Main function."""
    # Setup plotting style
    setup_plotting_style()
    
    # Create pipeline
    pipeline = AdversarialRobustnessPipeline(
        args.config,
        args.experiment_name
    )
    
    # Run pipeline
    if args.full_pipeline:
        pipeline.run_full_pipeline()
    else:
        # Run individual components
        if args.prepare_data:
            pipeline.prepare_data(use_lighting_correction=args.use_lighting_correction)
        
        if args.train_base:
            pipeline.train_base_model()
        
        if args.train_robust:
            pipeline.train_robust_model(use_pretrained=args.use_pretrained)
        
        if args.quantize:
            pipeline.quantize_model()
        
        if args.evaluate:
            pipeline.evaluate_robustness()
        
        if args.analyse_bias:
            pipeline.analyse_bias()
        
        if args.test_privacy:
            pipeline.test_privacy_preservation()
        
        if args.measure_efficiency:
            pipeline.measure_efficiency()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run adversarial robustness pipeline"
    )
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--experiment-name", type=str,
                        default=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        help="Name for this experiment")
    
    # Pipeline options
    parser.add_argument("--full-pipeline", action="store_true",
                        help="Run complete pipeline")
    parser.add_argument("--prepare-data", action="store_true",
                        help="Prepare data loaders")
    parser.add_argument("--train-base", action="store_true",
                        help="Train base model")
    parser.add_argument("--train-robust", action="store_true",
                        help="Train robust model")
    parser.add_argument("--quantize", action="store_true",
                        help="Quantize model")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate robustness")
    parser.add_argument("--analyse-bias", action="store_true",
                        help="Analyse bias")
    parser.add_argument("--test-privacy", action="store_true",
                        help="Test privacy preservation")
    parser.add_argument("--measure-efficiency", action="store_true",
                        help="Measure efficiency")
    
    # Additional options
    parser.add_argument("--use-lighting-correction", action="store_true",
                        help="Use lighting correction preprocessing")
    parser.add_argument("--use-pretrained", action="store_true",
                        help="Use pretrained model for robust training")
    
    args = parser.parse_args()
    
    # Default to full pipeline if no specific components selected
    if not any([args.prepare_data, args.train_base, args.train_robust,
                args.quantize, args.evaluate, args.analyse_bias,
                args.test_privacy, args.measure_efficiency]):
        args.full_pipeline = True
    
    main(args)