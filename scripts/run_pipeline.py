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
import pickle
import traceback

from src.analysis.enhanced_visualisation import setup_plotting_style
from src.utils.data_loader import load_config, create_data_loaders, get_class_names
from src.preprocessing.lighting_correction import create_lighting_corrected_transform
from src.preprocessing.data_augmentation import AugmentationWrapper
from src.models.base_classifier import create_model
from src.models.robust_classifier import create_robust_model_architecture

from src.analysis.enhanced_visualisation import ComprehensiveVisualiser
from src.defences.adversarial_training import AdversarialTrainer, create_robust_model

from src.models.quantized_model import quantize_robust_model
from src.attacks.fgsm import FGSMAttack
from src.attacks.pgd import PGDAttack
from src.analysis.bias_analysis import BiasAnalyser
from src.analysis.enhanced_visualisation import generate_final_report
from src.privacy.homomorphic_encryption import demonstrate_private_inference
from src.utils.metrics import RobustnessEvaluator, calculate_efficiency_metrics
from src.utils.training_utils import train_model


class PipelineCheckpoint:
    """Manages pipeline checkpoints for recovery."""
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.checkpoint_file = experiment_dir / 'pipeline_checkpoint.pkl'
        self.state_file = experiment_dir / 'pipeline_state.json'
    
    def save(self, stage: str, data: dict):
        """Save checkpoint for a pipeline stage."""
        # Save state
        state = {
            'last_completed_stage': stage,
            'timestamp': datetime.now().isoformat(),
            'stages_completed': self._get_completed_stages()
        }
        state['stages_completed'].append(stage)
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save stage-specific data
        stage_file = self.experiment_dir / f'{stage}_checkpoint.pkl'
        with open(stage_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Checkpoint saved for stage: {stage}")
    
    def load(self, stage: str) -> dict:
        """Load checkpoint for a specific stage."""
        stage_file = self.experiment_dir / f'{stage}_checkpoint.pkl'
        if stage_file.exists():
            with open(stage_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def get_last_stage(self) -> str:
        """Get the last completed stage."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                return state.get('last_completed_stage', None)
        return None
    
    def _get_completed_stages(self) -> list:
        """Get list of completed stages."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                return state.get('stages_completed', [])
        return []
    
    def can_skip_stage(self, stage: str) -> bool:
        """Check if a stage can be skipped based on checkpoints."""
        return stage in self._get_completed_stages()


class AdversarialRobustnessPipeline:
    """Complete pipeline for adversarially robust image classification."""
    
    def __init__(self, config_path: str, experiment_name: str, dev_mode: bool = False):
        """
        Initialise pipeline.
        
        Args:
            config_path: Path to configuration file
            experiment_name: Name for this experiment
            dev_mode: Enable development mode for faster iteration
        """
        self.config = load_config(config_path)
        self.experiment_name = experiment_name
        self.dev_mode = dev_mode
        
        # Ensure CUDA availability is properly checked
        if self.config['hardware']['device'] == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, using CPU")
            self.config['hardware']['device'] = 'cpu'
        
        self.device = torch.device(self.config['hardware']['device'])
        
        # Adjust config for dev mode
        if self.dev_mode:
            print("\nDEVELOPMENT MODE ENABLED - Using reduced settings for faster iteration")
            self.config['training']['epochs'] = min(2, self.config['training']['epochs'])
            self.config['adversarial']['attacks']['pgd']['num_steps'] = 5
            self.config['quantization']['calibration_batches'] = 10
            self.config['data']['batch_size'] = min(64, self.config['data']['batch_size'])
        
        # Create experiment directory
        self.experiment_dir = Path('experiments') / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize checkpoint manager
        self.checkpoint = PipelineCheckpoint(self.experiment_dir)
        
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
        # Check if we can skip this stage
        if self.checkpoint.can_skip_stage('data_preparation'):
            print("\nSkipping data preparation (checkpoint found)")
            checkpoint_data = self.checkpoint.load('data_preparation')
            if checkpoint_data:
                # Note: DataLoaders can't be pickled, so we'll recreate them
                print("  Recreating data loaders from checkpoint...")
            # Still need to create the loaders
        
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
        
        # Save checkpoint
        self.checkpoint.save('data_preparation', {
            'dataset': self.config['data']['dataset_name'],
            'use_lighting_correction': use_lighting_correction,
            'num_train': len(self.train_loader.dataset),
            'num_val': len(self.val_loader.dataset),
            'num_test': len(self.test_loader.dataset)
        })
    
    def train_base_model(self):
        """Train base model without adversarial training."""
        # Check for existing checkpoint
        model_path = self.experiment_dir / 'base_model.pth'
        if model_path.exists() and self.checkpoint.can_skip_stage('base_training'):
            print("\nLoading base model from checkpoint")
            self.model = create_model(self.config)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            return
        
        print("\n" + "="*50)
        print("BASE MODEL TRAINING")
        print("="*50)
        
        # Create model and ensure it's on the correct device
        self.model = create_model(self.config)
        self.model = self.model.to(self.device)
        
        print(f"Architecture: {self.config['model']['architecture']}")
        print(f"Parameters: {self.model.get_num_parameters():,}")
        print(f"Device: {next(self.model.parameters()).device}")
        
        # Create augmentation wrapper
        augmentation = AugmentationWrapper(
            self.config,
            use_mixup=True,
            use_cutmix=True,
            use_autoaugment=not self.dev_mode  # Skip in dev mode for speed
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
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, model_path)
        print(f"Base model saved to: {model_path}")
        
        # Save checkpoint
        self.checkpoint.save('base_training', {
            'model_path': str(model_path),
            'architecture': self.config['model']['architecture'],
            'parameters': self.model.get_num_parameters()
        })
    
    def train_robust_model(self, use_pretrained: bool = True):
        """Train adversarially robust model."""
        # Check for existing checkpoint
        robust_path = self.experiment_dir / 'robust_model.pth'
        if robust_path.exists() and self.checkpoint.can_skip_stage('adversarial_training'):
            print("\nLoading robust model from checkpoint")
            if self.model is None:
                self.model = create_model(self.config)
            self.robust_model = self.model
            checkpoint = torch.load(robust_path, map_location=self.device)
            self.robust_model.load_state_dict(checkpoint['model_state_dict'])
            self.robust_model.to(self.device)
            return
        
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
        
        # Ensure model is on correct device
        self.robust_model = self.robust_model.to(self.device)
        
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
        print(f"  - Device: {next(self.robust_model.parameters()).device}")
        if self.dev_mode:
            print(f"  - DEV MODE: Reduced epochs and attack steps")
        
        # Train
        trainer.train(
            self.train_loader,
            self.val_loader,
            save_path=robust_path
        )
        
        self.robust_model = trainer.model
        
        # Save checkpoint
        self.checkpoint.save('adversarial_training', {
            'model_path': str(robust_path),
            'training_history': trainer.history
        })
    
    def quantize_model(self):
        """Quantize the robust model."""
        # Check for existing checkpoint
        quantized_path = self.experiment_dir / 'quantized_model.pth'
        if quantized_path.exists() and self.checkpoint.can_skip_stage('quantization'):
            print("\nSkipping quantization (checkpoint found)")
            # Still need to load for later stages
            if self.robust_model is None:
                self.robust_model = create_model(self.config)
                robust_path = self.experiment_dir / 'robust_model.pth'
                if robust_path.exists():
                    checkpoint = torch.load(robust_path, map_location=self.device)
                    self.robust_model.load_state_dict(checkpoint['model_state_dict'])
            return
        
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
        
        try:
            # Quantize
            self.quantized_model = quantize_robust_model(
                self.robust_model,
                self.config,
                calibration_loader=calibration_loader if self.config['quantization']['type'] == 'static' else None,
                test_loader=self.test_loader,
                device=self.device,
                save_path=quantized_path
            )
            
            # Save checkpoint
            self.checkpoint.save('quantization', {
                'model_path': str(quantized_path),
                'quantization_type': self.config['quantization']['type']
            })
            
        except Exception as e:
            print(f"\nQuantization failed: {e}")
            print("Continuing with non-quantized model...")
            self.quantized_model = self.robust_model
            # Still save checkpoint to mark stage as complete
            self.checkpoint.save('quantization', {
                'error': str(e),
                'fallback': 'using_robust_model'
            })
    
    
    
    
    
    
    def evaluate_robustness(self):
        """Comprehensive robustness evaluation."""
        # Check if we can skip or load cached results
        if self.checkpoint.can_skip_stage('robustness_evaluation') and not self.dev_mode:
            print("\nLoading robustness evaluation from checkpoint")
            checkpoint_data = self.checkpoint.load('robustness_evaluation')
            if checkpoint_data and 'results' in checkpoint_data:
                return checkpoint_data['results']
        
        print("\n" + "="*50)
        print("ROBUSTNESS EVALUATION")
        print("="*50)
        
        models_to_evaluate = []
        
        # Ensure models are on correct device
        if self.model is not None:
            self.model = self.model.to(self.device)
            models_to_evaluate.append(('Base Model', self.model))
        if self.robust_model is not None:
            self.robust_model = self.robust_model.to(self.device)
            models_to_evaluate.append(('Robust Model', self.robust_model))
        if self.quantized_model is not None and self.quantized_model != self.robust_model:
            # IMPORTANT: Keep quantized model on CPU
            models_to_evaluate.append(('Quantized Model', self.quantized_model.cpu()))
        
        # Create attacks (with reduced parameters in dev mode)
        if self.dev_mode:
            attacks = [
                FGSMAttack(epsilon=0.03),
                PGDAttack(
                    epsilon=0.03,
                    alpha=0.01,
                    num_steps=5  # Reduced for dev mode
                )
            ]
        else:
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
            
            # Determine device for evaluation
            if 'Quantized' in model_name:
                eval_device = torch.device('cpu')
                # Create CPU data loader for quantized model
                if self.dev_mode:
                    subset_size = min(500, len(self.test_loader.dataset))
                    subset_indices = torch.randperm(len(self.test_loader.dataset))[:subset_size]
                    subset = torch.utils.data.Subset(self.test_loader.dataset, subset_indices)
                    eval_loader = torch.utils.data.DataLoader(
                        subset,
                        batch_size=self.config['data']['batch_size'],
                        shuffle=False,
                        num_workers=0  # Use 0 workers for CPU evaluation
                    )
                else:
                    eval_loader = torch.utils.data.DataLoader(
                        self.test_loader.dataset,
                        batch_size=self.config['data']['batch_size'],
                        shuffle=False,
                        num_workers=0
                    )
                
                # Evaluate quantized model on CPU
                evaluator = RobustnessEvaluator(model, self.config)
                evaluator.device = eval_device  # Force CPU for quantized model
                eval_results = evaluator.evaluate_comprehensive(eval_loader, attacks)
            else:
                # Regular evaluation on GPU
                evaluator = RobustnessEvaluator(model, self.config)
                
                # In dev mode, evaluate on subset
                if self.dev_mode:
                    subset_size = min(500, len(self.test_loader.dataset))
                    subset_indices = torch.randperm(len(self.test_loader.dataset))[:subset_size]
                    subset = torch.utils.data.Subset(self.test_loader.dataset, subset_indices)
                    subset_loader = torch.utils.data.DataLoader(
                        subset,
                        batch_size=self.config['data']['batch_size'],
                        shuffle=False
                    )
                    eval_results = evaluator.evaluate_comprehensive(subset_loader, attacks)
                else:
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
            
            # Handle potential missing data
            fgsm_data = eval_results[
                (eval_results['attack'] == 'FGSMAttack') & 
                (eval_results['epsilon'] == 0.03)
            ]
            fgsm_acc = fgsm_data['accuracy'].values[0] if len(fgsm_data) > 0 else 0
            
            pgd_data = eval_results[
                (eval_results['attack'] == 'PGDAttack') & 
                (eval_results['epsilon'] == 0.03)
            ]
            pgd_acc = pgd_data['accuracy'].values[0] if len(pgd_data) > 0 else 0
            
            print(f"  Clean accuracy: {clean_acc:.2f}%")
            print(f"  FGSM robustness: {fgsm_acc:.2f}%")
            print(f"  PGD robustness: {pgd_acc:.2f}%")
        
        # Save checkpoint
        self.checkpoint.save('robustness_evaluation', {
            'results': results,
            'dev_mode': self.dev_mode
        })
        
        return results
    
    
    
    
    
    
    
    def analyse_bias(self):
        """Perform bias analysis."""
        # Check for checkpoint
        if self.checkpoint.can_skip_stage('bias_analysis') and not self.dev_mode:
            print("\nLoading bias analysis from checkpoint")
            checkpoint_data = self.checkpoint.load('bias_analysis')
            if checkpoint_data and 'bias_report' in checkpoint_data:
                return checkpoint_data['bias_report']
        
        print("\n" + "="*50)
        print("BIAS ANALYSIS")
        print("="*50)
        
        if self.robust_model is None:
            print("No model available for bias analysis!")
            return None
        
        # Ensure model is on correct device
        self.robust_model = self.robust_model.to(self.device)
        
        # Get class names
        class_names = get_class_names(self.config['data']['dataset_name'])
        
        # Create analyser
        analyser = BiasAnalyser(self.config, class_names)
        
        # In dev mode, use subset
        if self.dev_mode:
            print("  Using subset for faster analysis...")
            # Create small subset loaders
            train_subset = torch.utils.data.Subset(
                self.train_loader.dataset,
                torch.randperm(len(self.train_loader.dataset))[:1000]
            )
            test_subset = torch.utils.data.Subset(
                self.test_loader.dataset,
                torch.randperm(len(self.test_loader.dataset))[:500]
            )
            
            train_subset_loader = torch.utils.data.DataLoader(
                train_subset, batch_size=self.config['data']['batch_size']
            )
            test_subset_loader = torch.utils.data.DataLoader(
                test_subset, batch_size=self.config['data']['batch_size']
            )
            
            bias_report = analyser.generate_bias_report(
                self.robust_model,
                train_subset_loader,
                test_subset_loader
            )
        else:
            # Generate full bias report
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
        
        # Save checkpoint
        self.checkpoint.save('bias_analysis', {
            'bias_report': bias_report,
            'patterns': patterns,
            'dev_mode': self.dev_mode
        })
        
        return bias_report
    
    def test_privacy_preservation(self):
        """Test homomorphic encryption for privacy-preserving inference."""
        # Skip in dev mode by default
        if self.dev_mode:
            print("\nSkipping privacy testing in dev mode")
            return
        
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
            
            # Save checkpoint
            self.checkpoint.save('privacy_test', results)
        
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
        if self.quantized_model is not None and self.quantized_model != self.robust_model:
            models_to_analyse.append(('Quantized Model', self.quantized_model))
        
        # Input shape
        sample_data, _ = next(iter(self.test_loader))
        input_shape = sample_data[0].shape
        
        efficiency_results = {}
        
        for model_name, model in models_to_analyse:
            print(f"\nAnalysing {model_name}...")
            
            # Determine device for model
            if hasattr(model, 'qconfig') or 'Quantized' in model_name:
                # Quantized models should be evaluated on CPU
                eval_device = torch.device('cpu')
                model = model.cpu()
            else:
                eval_device = self.device
                model = model.to(eval_device)
            
            # Reduce runs in dev mode
            num_runs = 10 if self.dev_mode else 100
            
            metrics = calculate_efficiency_metrics(
                model,
                (1, *input_shape),
                eval_device,
                num_runs=num_runs
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
        
        # Save checkpoint
        self.checkpoint.save('efficiency_analysis', efficiency_results)
        
        return efficiency_results
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def generate_report(self, evaluation_results, bias_report):
        """Generate comprehensive report with actual visualizations."""
        print("\n" + "="*50)
        print("GENERATING REPORT")
        print("="*50)
        
        # Create visualizer
        visualizer = ComprehensiveVisualiser(
            str(self.experiment_dir),
            self.experiment_name
        )
        
        # Load training history if available
        training_history = {}
        checkpoint_data = self.checkpoint.load('adversarial_training')
        if checkpoint_data and 'training_history' in checkpoint_data:
            training_history = checkpoint_data['training_history']
        elif hasattr(self, 'robust_model') and hasattr(self.robust_model, 'history'):
            training_history = self.robust_model.history
        
        # Generate all visualizations
        print("\nGenerating visualizations...")
        visualizer.generate_all_visualisations(
            evaluation_results,
            bias_report,
            training_history,
            self.config
        )
        
        # Also generate simple matplotlib plots for quick viewing
        self._generate_simple_plots(evaluation_results, bias_report)
        
        # Create HTML report
        from src.analysis.enhanced_visualisation import create_model_report
        
        # Prepare evaluation summary
        eval_summary = {}
        if evaluation_results and 'Robust Model' in evaluation_results:
            results_df = evaluation_results['Robust Model']
            eval_summary['clean_accuracy'] = results_df[
                results_df['attack'] == 'None'
            ]['accuracy'].values[0]
            
            fgsm_results = results_df[
                (results_df['attack'] == 'FGSMAttack') & 
                (results_df['epsilon'].abs() - 0.03).abs() < 0.001
            ]
            if not fgsm_results.empty:
                eval_summary['fgsm_accuracy'] = fgsm_results['accuracy'].values[0]
            
            pgd_results = results_df[
                (results_df['attack'] == 'PGDAttack') & 
                (results_df['epsilon'].abs() - 0.03).abs() < 0.001
            ]
            if not pgd_results.empty:
                eval_summary['pgd_accuracy'] = pgd_results['accuracy'].values[0]
        
        # Create report
        create_model_report(
            self.config,
            eval_summary,
            bias_report,
            str(self.experiment_dir)
        )
        
        print(f"\n Report and visualizations generated in: {self.experiment_dir}")
        print(f"    Figures: {self.experiment_dir}/figures/")
        print(f"    HTML Report: {self.experiment_dir}/evaluation_report.html")
        print(f"    Interactive Dashboard: {self.experiment_dir}/figures/summary_dashboard.html")
        
        # Save final checkpoint
        self.checkpoint.save('report_generation', {
            'report_dir': str(self.experiment_dir),
            'timestamp': datetime.now().isoformat()
        })

    def _generate_simple_plots(self, evaluation_results, bias_report):
        """Generate simple matplotlib plots for immediate viewing."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        figures_dir = self.experiment_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        # 1. Model Comparison Bar Chart
        if evaluation_results:
            plt.figure(figsize=(10, 6))
            
            models = []
            clean_accs = []
            robust_accs = []
            
            for model_name, results_df in evaluation_results.items():
                models.append(model_name)
                clean_acc = results_df[results_df['attack'] == 'None']['accuracy'].values[0]
                clean_accs.append(clean_acc)
                
                # Get PGD accuracy
                pgd_data = results_df[
                    (results_df['attack'] == 'PGDAttack') & 
                    (results_df['epsilon'] == 0.03)
                ]
                robust_acc = pgd_data['accuracy'].values[0] if not pgd_data.empty else 0
                robust_accs.append(robust_acc)
            
            x = range(len(models))
            width = 0.35
            
            plt.bar([i - width/2 for i in x], clean_accs, width, label='Clean Accuracy', color='skyblue')
            plt.bar([i + width/2 for i in x], robust_accs, width, label='Robust Accuracy (PGD)', color='coral')
            
            plt.xlabel('Model')
            plt.ylabel('Accuracy (%)')
            plt.title('Model Performance Comparison')
            plt.xticks(x, models)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(figures_dir / 'model_comparison.png', dpi=150)
            plt.close()
            print("   ✓ Generated model_comparison.png")
        
        # 2. Training History Plot
        if hasattr(self, 'checkpoint'):
            checkpoint_data = self.checkpoint.load('adversarial_training')
            if checkpoint_data and 'training_history' in checkpoint_data:
                history = checkpoint_data['training_history']
                
                plt.figure(figsize=(12, 5))
                
                # Loss subplot
                plt.subplot(1, 2, 1)
                if 'train_loss' in history:
                    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
                if 'val_loss' in history:
                    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Accuracy subplot
                plt.subplot(1, 2, 2)
                if 'train_acc' in history:
                    plt.plot(history['train_acc'], label='Train Acc', linewidth=2)
                if 'val_acc' in history:
                    plt.plot(history['val_acc'], label='Val Acc', linewidth=2)
                if 'val_robust_acc' in history:
                    plt.plot(history['val_robust_acc'], label='Robust Acc', linewidth=2, linestyle='--')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy (%)')
                plt.title('Training Progress')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(figures_dir / 'training_history.png', dpi=150)
                plt.close()
                print("   ✓ Generated training_history.png")
        
        # 3. Robustness Curves
        if evaluation_results and 'Robust Model' in evaluation_results:
            results_df = evaluation_results['Robust Model']
            
            plt.figure(figsize=(10, 6))
            
            for attack in ['FGSMAttack', 'PGDAttack']:
                attack_data = results_df[results_df['attack'] == attack]
                if not attack_data.empty:
                    plt.plot(attack_data['epsilon'], attack_data['accuracy'], 
                            marker='o', linewidth=2, markersize=8, 
                            label=attack.replace('Attack', ''))
            
            # Add clean accuracy line
            clean_acc = results_df[results_df['attack'] == 'None']['accuracy'].values[0]
            plt.axhline(y=clean_acc, color='green', linestyle='--', label='Clean Accuracy')
            
            plt.xlabel('Epsilon (ε)')
            plt.ylabel('Accuracy (%)')
            plt.title('Model Robustness Against Adversarial Attacks')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xlim(0, max(results_df['epsilon']) * 1.1)
            
            plt.tight_layout()
            plt.savefig(figures_dir / 'robustness_curves.png', dpi=150)
            plt.close()
            print("   ✓ Generated robustness_curves.png")
        
        # 4. Class Performance from Bias Report
        if bias_report is not None:
            # Filter out summary rows
            class_data = bias_report[~bias_report['Class'].isin(['Overall', 'Std Dev'])]
            
            if not class_data.empty:
                plt.figure(figsize=(12, 6))
                
                # Create bar plot
                x = range(len(class_data))
                plt.bar(x, class_data['Accuracy'], color='lightblue', label='Accuracy')
                
                # Add robust accuracy if available
                if 'PGD Robustness' in class_data.columns:
                    plt.bar(x, class_data['PGD Robustness'], alpha=0.7, color='orange', label='PGD Robustness')
                
                plt.xlabel('Class')
                plt.ylabel('Performance (%)')
                plt.title('Per-Class Performance Analysis')
                plt.xticks(x, class_data['Class'], rotation=45)
                plt.legend()
                plt.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                plt.savefig(figures_dir / 'class_performance.png', dpi=150)
                plt.close()
                print("   ✓ Generated class_performance.png")
        
        # 5. Quick summary plot
        self._generate_summary_plot(evaluation_results)




        def _generate_summary_plot(self, evaluation_results):
            """Generate a summary plot showing key metrics."""
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            
            figures_dir = self.experiment_dir / 'figures'
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Title and experiment info
            ax.text(0.5, 0.95, f'Adversarial Robustness Pipeline Summary', 
                    ha='center', va='top', fontsize=20, fontweight='bold', transform=ax.transAxes)
            ax.text(0.5, 0.90, f'Experiment: {self.experiment_name}', 
                    ha='center', va='top', fontsize=14, style='italic', transform=ax.transAxes)
            
            # Key metrics boxes
            y_pos = 0.75
            box_height = 0.12
            box_spacing = 0.02
            
            # Dataset info
            ax.add_patch(Rectangle((0.1, y_pos), 0.35, box_height, 
                                facecolor='lightblue', edgecolor='black', linewidth=2))
            ax.text(0.275, y_pos + box_height/2, f"Dataset: {self.config['data']['dataset_name']}\n"
                    f"Architecture: {self.config['model']['architecture']}", 
                    ha='center', va='center', fontsize=12, fontweight='bold')
            
            # Training info
            ax.add_patch(Rectangle((0.55, y_pos), 0.35, box_height, 
                                facecolor='lightgreen', edgecolor='black', linewidth=2))
            ax.text(0.725, y_pos + box_height/2, f"Epochs: {self.config['training']['epochs']}\n"
                    f"Adv Training: {'✓' if self.config['adversarial']['training']['enable'] else '✗'}", 
                    ha='center', va='center', fontsize=12, fontweight='bold')
            
            # Results
            if evaluation_results and 'Robust Model' in evaluation_results:
                results_df = evaluation_results['Robust Model']
                clean_acc = results_df[results_df['attack'] == 'None']['accuracy'].values[0]
                
                pgd_data = results_df[(results_df['attack'] == 'PGDAttack') & (results_df['epsilon'] == 0.03)]
                robust_acc = pgd_data['accuracy'].values[0] if not pgd_data.empty else 0
                
                y_pos -= (box_height + box_spacing * 2)
                
                # Clean accuracy box
                color = 'lightcoral' if clean_acc < 50 else 'lightgreen'
                ax.add_patch(Rectangle((0.1, y_pos), 0.35, box_height, 
                                    facecolor=color, edgecolor='black', linewidth=2))
                ax.text(0.275, y_pos + box_height/2, f"Clean Accuracy\n{clean_acc:.1f}%", 
                        ha='center', va='center', fontsize=14, fontweight='bold')
                
                # Robust accuracy box
                color = 'lightcoral' if robust_acc < 40 else 'lightyellow' if robust_acc < 60 else 'lightgreen'
                ax.add_patch(Rectangle((0.55, y_pos), 0.35, box_height, 
                                    facecolor=color, edgecolor='black', linewidth=2))
                ax.text(0.725, y_pos + box_height/2, f"Robust Accuracy\n{robust_acc:.1f}%", 
                        ha='center', va='center', fontsize=14, fontweight='bold')
            
            # Status message
            ax.text(0.5, 0.1, ' Pipeline Completed Successfully', 
                    ha='center', va='bottom', fontsize=16, color='green', fontweight='bold', 
                    transform=ax.transAxes)
            
            # Remove axes
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(figures_dir / 'pipeline_summary.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("   ✓ Generated pipeline_summary.png")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def run_full_pipeline(self, resume: bool = False):
        """Run the complete pipeline."""
        start_time = time.time()
        
        print("\n" + "="*60)
        print("ADVERSARIAL ROBUSTNESS PIPELINE")
        print("="*60)
        print(f"Experiment: {self.experiment_name}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.dev_mode:
            print("Running in DEVELOPMENT MODE")
        if resume:
            last_stage = self.checkpoint.get_last_stage()
            if last_stage:
                print(f"Resuming from checkpoint (last stage: {last_stage})")
        
        # Track which stages to run
        stages = [
            ('data_preparation', lambda: self.prepare_data(use_lighting_correction=True)),
            ('base_training', lambda: self.train_base_model() if not self.config.get('skip_base_training', False) else None),
            ('adversarial_training', lambda: self.train_robust_model(use_pretrained=True)),
            ('quantization', lambda: self.quantize_model() if self.config['quantization']['enable'] else None),
            ('robustness_evaluation', lambda: self.evaluate_robustness()),
            ('bias_analysis', lambda: self.analyse_bias()),
            ('privacy_test', lambda: self.test_privacy_preservation()),
            ('efficiency_analysis', lambda: self.measure_efficiency()),
        ]
        
        evaluation_results = None
        bias_report = None
        
        # Run stages with error handling
        for stage_name, stage_fn in stages:
            try:
                if stage_name == 'robustness_evaluation':
                    evaluation_results = stage_fn()
                elif stage_name == 'bias_analysis':
                    bias_report = stage_fn()
                else:
                    stage_fn()
                    
            except Exception as e:
                print(f"\nError in stage '{stage_name}': {str(e)}")
                print(f"Traceback:\n{traceback.format_exc()}")
                
                # Ask user if they want to continue
                if not self.dev_mode:
                    response = input("\nContinue with next stage? (y/n): ")
                    if response.lower() != 'y':
                        print("Pipeline aborted.")
                        return
                else:
                    print("Continuing to next stage...")
        
        # Generate report
        try:
            self.generate_report(evaluation_results, bias_report)
        except Exception as e:
            print(f"\nError generating report: {str(e)}")
        
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
            'quantization': self.config['quantization']['enable'],
            'dev_mode': self.dev_mode,
            'completed_stages': self.checkpoint._get_completed_stages()
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
        args.experiment_name,
        dev_mode=args.dev_mode
    )
    
    # Run pipeline
    if args.full_pipeline:
        pipeline.run_full_pipeline(resume=args.resume)
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
    parser.add_argument("--dev-mode", action="store_true",
                        help="Enable development mode (faster iteration with reduced settings)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if available")
    
    args = parser.parse_args()
    
    # Default to full pipeline if no specific components selected
    if not any([args.prepare_data, args.train_base, args.train_robust,
                args.quantize, args.evaluate, args.analyse_bias,
                args.test_privacy, args.measure_efficiency]):
        args.full_pipeline = True
    
    main(args)