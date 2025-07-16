# Adversarial training implementation for robust models
# src/defences/adversarial_training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable, Union
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

from ..attacks.fgsm import FGSMAttack
from ..attacks.pgd import PGDAttack
from ..utils.metrics import MetricsTracker
from ..utils.training_utils import LearningRateScheduler, TrainingLogger, EarlyStopping


class AdversarialTrainer:
    """Trainer for adversarially robust models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device
    ):
        """
        Initialise adversarial trainer.
        
        Args:
            model: Model to train
            config: Configuration dictionary
            device: Device to use
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.adv_config = config['adversarial']['training']
        
        # Training parameters
        self.epochs = config['training']['epochs']
        self.attack_ratio = self.adv_config['attack_ratio']
        self.epsilon_schedule = self.adv_config['epsilon_schedule']
        self.max_epsilon = self.adv_config['max_epsilon']
        
        # Create attacks
        self.fgsm = FGSMAttack(epsilon=self.max_epsilon)
        self.pgd = PGDAttack(
            epsilon=self.max_epsilon,
            alpha=self.max_epsilon / 4,
            num_steps=7,  # Reduced for training efficiency
            random_start=True
        )
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=0.9,
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = LearningRateScheduler(self.optimizer, config['training']['scheduler'])
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_robust_acc': [],
            'learning_rate': []
        }
    
    def get_current_epsilon(self, epoch: int) -> float:
        """Get epsilon value for current epoch based on schedule."""
        if self.epsilon_schedule == 'constant':
            return self.max_epsilon
        elif self.epsilon_schedule == 'linear':
            # Linear increase from 0 to max_epsilon
            return self.max_epsilon * (epoch / self.epochs)
        elif self.epsilon_schedule == 'exponential':
            # Exponential increase
            return self.max_epsilon * (1 - np.exp(-5 * epoch / self.epochs))
        else:
            return self.max_epsilon
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch with adversarial examples."""
        self.model.train()
        tracker = MetricsTracker(self.config['model']['num_classes'])
        total_loss = 0
        num_batches = len(train_loader)
        
        # Get current epsilon
        current_epsilon = self.get_current_epsilon(epoch)
        self.fgsm.epsilon = current_epsilon
        self.pgd.epsilon = current_epsilon
        self.pgd.alpha = current_epsilon / 4
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} (ε={current_epsilon:.3f})')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Determine if this batch should use adversarial training
            use_adversarial = np.random.rand() < self.attack_ratio
            
            if use_adversarial and current_epsilon > 0:
                # Generate adversarial examples
                # Use PGD 70% of the time, FGSM 30%
                if np.random.rand() < 0.7:
                    adv_data = self.pgd.generate(self.model, data, target)
                else:
                    adv_data = self.fgsm.generate(self.model, data, target)
                
                # Mix clean and adversarial data
                mixed_data = torch.cat([data, adv_data])
                mixed_target = torch.cat([target, target])
            else:
                mixed_data = data
                mixed_target = target
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(mixed_data)
            loss = self.criterion(outputs, mixed_target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Update metrics (on clean data only for accurate tracking)
            with torch.no_grad():
                clean_outputs = self.model(data)
                predictions = clean_outputs.argmax(dim=1)
                probabilities = torch.softmax(clean_outputs, dim=1)
                tracker.update(predictions, target, probabilities, loss.item())
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        # Compute metrics
        metrics = tracker.compute_metrics()
        metrics['loss'] = total_loss / num_batches
        metrics['epsilon'] = current_epsilon
        
        return metrics
    
    def validate(
        self,
        val_loader: DataLoader,
        test_robustness: bool = True
    ) -> Dict[str, float]:
        """Validate model with optional robustness testing."""
        self.model.eval()
        
        # Clean accuracy
        clean_tracker = MetricsTracker(self.config['model']['num_classes'])
        total_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                
                predictions = outputs.argmax(dim=1)
                probabilities = torch.softmax(outputs, dim=1)
                clean_tracker.update(predictions, target, probabilities, loss.item())
                total_loss += loss.item()
        
        clean_metrics = clean_tracker.compute_metrics()
        clean_metrics['loss'] = total_loss / len(val_loader)
        
        # Robust accuracy (on subset for efficiency)
        robust_acc = 0
        if test_robustness and self.max_epsilon > 0:
            num_robust_batches = min(10, len(val_loader))  # Test on subset
            robust_correct = 0
            robust_total = 0
            
            for i, (data, target) in enumerate(val_loader):
                if i >= num_robust_batches:
                    break
                
                data, target = data.to(self.device), target.to(self.device)
                
                # Test with PGD
                adv_data = self.pgd.generate(self.model, data, target)
                
                with torch.no_grad():
                    adv_outputs = self.model(adv_data)
                    adv_preds = adv_outputs.argmax(dim=1)
                    robust_correct += adv_preds.eq(target).sum().item()
                    robust_total += target.size(0)
            
            robust_acc = 100. * robust_correct / robust_total if robust_total > 0 else 0
        
        return {
            'loss': clean_metrics['loss'],
            'accuracy': clean_metrics['accuracy'],
            'robust_accuracy': robust_acc
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: Optional[Path] = None
    ):
        """Complete adversarial training loop."""
        print(f"\nStarting adversarial training...")
        print(f"Attack ratio: {self.attack_ratio}")
        print(f"Epsilon schedule: {self.epsilon_schedule}")
        print(f"Max epsilon: {self.max_epsilon}")
        
        best_val_acc = 0
        early_stopping = EarlyStopping(patience=10, mode='max')
        
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, test_robustness=(epoch % 5 == 0))
            
            # Update scheduler
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.scheduler.get_lr()
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_robust_acc'].append(val_metrics['robust_accuracy'])
            self.history['learning_rate'].append(current_lr)
            
            epoch_time = time.time() - start_time
            
            # Print progress
            print(f"\nEpoch {epoch}/{self.epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            if val_metrics['robust_accuracy'] > 0:
                print(f"Robust Acc: {val_metrics['robust_accuracy']:.2f}%")
            print(f"Epsilon: {train_metrics['epsilon']:.4f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_acc': best_val_acc,
                        'config': self.config
                    }, save_path)
                    print(f"Saved best model (acc: {best_val_acc:.2f}%)")
            
            # Early stopping
            if early_stopping(val_metrics['accuracy']):
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")


def create_robust_model(
    model: nn.Module,
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_path: Optional[Path] = None,
    use_free_training: bool = False
) -> nn.Module:
    """
    Create adversarially robust model through training.
    
    Args:
        model: Base model to train
        config: Configuration dictionary
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to use
        save_path: Path to save model
        use_free_training: Whether to use free adversarial training
    
    Returns:
        Trained robust model
    """
    if use_free_training:
        print("Note: Free adversarial training not implemented, using standard adversarial training")
    
    trainer = AdversarialTrainer(model, config, device)
    trainer.train(train_loader, val_loader, save_path)
    
    return trainer.model