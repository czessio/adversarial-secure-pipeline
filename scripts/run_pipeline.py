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
        
        print(f"✓ Checkpoint saved for stage: {stage}")
    
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
        self.device = torch.device(
            self.config['hardware']['device'] if torch.cuda.is_available() else 'cpu'
        )
        
        # Adjust config for dev mode
        if self.dev_mode:
            print("\n DEVELOPMENT MODE ENABLED - Using reduced settings for faster iteration")
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
            print("\n✓ Skipping data preparation (checkpoint found)")
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
            print("\n✓ Loading base model from checkpoint")
            self.model = create_model(self.config)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            return
        
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
            print("\n✓ Loading robust model from checkpoint")
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
            print("\n✓ Skipping quantization (checkpoint found)")
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
            print(f"\n⚠️  Quantization failed: {e}")
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
            print("\n✓ Loading robustness evaluation from checkpoint")
            checkpoint_data = self.checkpoint.load('robustness_evaluation')
            if checkpoint_data and 'results' in checkpoint_data:
                return checkpoint_data['results']
        
        print("\n" + "="*50)
        print("ROBUSTNESS EVALUATION")
        print("="*50)
        
        models_to_evaluate = []
        
        if self.model is not None:
            models_to_evaluate.append(('Base Model', self.model))
        if self.robust_model is not None:
            models_to_evaluate.append(('Robust Model', self.robust_model))
        if self.quantized_model is not None and self.quantized_model != self.robust_model:
            models_to_evaluate.append(('Quantized Model', self.quantized_model))
        
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
            
            evaluator = RobustnessEvaluator(model, self.config)
            
            # In dev mode, evaluate on subset
            if self.dev_mode:
                # Create subset loader
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
            fgsm_acc = eval_results[
                (eval_results['attack'] == 'FGSMAttack') & 
                (eval_results['epsilon'] == 0.03)
            ]['accuracy'].values[0]
            pgd_acc = eval_results[
                (eval_results['attack'] == 'PGDAttack') & 
                (eval_results['epsilon'] == 0.03)
            ]['accuracy'].values[0]
            
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
            print("\n✓ Loading bias analysis from checkpoint")
            checkpoint_data = self.checkpoint.load('bias_analysis')
            if checkpoint_data and 'bias_report' in checkpoint_data:
                return checkpoint_data['bias_report']
        
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
            print("\n⚡ Skipping privacy testing in dev mode")
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
            
            # Reduce runs in dev mode
            num_runs = 10 if self.dev_mode else 100
            
            metrics = calculate_efficiency_metrics(
                model,
                (1, *input_shape),
                self.device,
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
        """Generate comprehensive report."""
        print("\n" + "="*50)
        print("GENERATING REPORT")
        print("="*50)
        
        # Prepare evaluation summary
        eval_summary = {}
        if evaluation_results and 'Robust Model' in evaluation_results:
            results_df = evaluation_results['Robust Model']
            eval_summary['clean_accuracy'] = results_df[
                results_df['attack'] == 'None'
            ]['accuracy'].values[0]
            
            # Find FGSM and PGD results at epsilon=0.03
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
        
        print(f"Report generated in: {self.experiment_dir}")
        
        # Save final checkpoint
        self.checkpoint.save('report_generation', {
            'report_dir': str(self.experiment_dir),
            'timestamp': datetime.now().isoformat()
        })
    
    def run_full_pipeline(self, resume: bool = False):
        """Run the complete pipeline."""
        start_time = time.time()
        
        print("\n" + "="*60)
        print("ADVERSARIAL ROBUSTNESS PIPELINE")
        print("="*60)
        print(f"Experiment: {self.experiment_name}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.dev_mode:
            print("🚀 Running in DEVELOPMENT MODE")
        if resume:
            last_stage = self.checkpoint.get_last_stage()
            if last_stage:
                print(f"📌 Resuming from checkpoint (last stage: {last_stage})")
        
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
                print(f"\n❌ Error in stage '{stage_name}': {str(e)}")
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
            print(f"\n❌ Error generating report: {str(e)}")
        
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