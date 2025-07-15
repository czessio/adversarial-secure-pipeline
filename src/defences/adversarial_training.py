# Adversarial training implementation for robust model training

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable, Tuple
import numpy as np
from tqdm import tqdm

from ..attacks.fgsm import FGSMAttack
from ..attacks.pgd import PGDAttack


class AdversarialTrainer:
    """Trainer for adversarial training of neural networks."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialise adversarial trainer.
        
        Args:
            model: Model to train
            config: Configuration dictionary
            device: Device to use for training
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Training parameters
        self.epochs = config['training']['epochs']
        self.lr = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']
        
        # Adversarial parameters
        self.adv_config = config['adversarial']
        self.attack_ratio = self.adv_config['training']['attack_ratio']
        self.epsilon_schedule = self.adv_config['training']['epsilon_schedule']
        self.max_epsilon = self.adv_config['training']['max_epsilon']
        
        # Initialise optimiser
        self.optimiser = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Initialise scheduler
        self._init_scheduler()
        
        # Initialise attacks
        self._init_attacks()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_robust_acc': []
        }
    
    def _init_scheduler(self):
        """Initialise learning rate scheduler."""
        scheduler_config = self.config['training']['scheduler']
        
        if scheduler_config['type'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimiser,
                T_max=self.epochs
            )
        elif scheduler_config['type'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimiser,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        elif scheduler_config['type'] == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimiser,
                gamma=scheduler_config['gamma']
            )
        else:
            self.scheduler = None
    
    def _init_attacks(self):
        """Initialise adversarial attacks."""
        fgsm_config = self.adv_config['attacks']['fgsm']
        pgd_config = self.adv_config['attacks']['pgd']
        
        self.fgsm = FGSMAttack(
            epsilon=fgsm_config['epsilon'],
            targeted=fgsm_config['targeted']
        )
        
        self.pgd = PGDAttack(
            epsilon=pgd_config['epsilon'],
            alpha=pgd_config['alpha'],
            num_steps=pgd_config['num_steps'],
            random_start=pgd_config['random_start']
        )
    
    def get_epsilon(self, epoch: int) -> float:
        """
        Get epsilon value for current epoch based on schedule.
        
        Args:
            epoch: Current epoch
        
        Returns:
            Epsilon value
        """
        if self.epsilon_schedule == 'constant':
            return self.max_epsilon
        elif self.epsilon_schedule == 'linear':
            return self.max_epsilon * (epoch + 1) / self.epochs
        elif self.epsilon_schedule == 'exponential':
            return self.max_epsilon * (1 - np.exp(-5 * epoch / self.epochs))
        else:
            return self.max_epsilon
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        criterion: nn.Module = nn.CrossEntropyLoss()
    ) -> Tuple[float, float]:
        """
        Train for one epoch with adversarial training.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch
            criterion: Loss function
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        # Update epsilon for current epoch
        current_epsilon = self.get_epsilon(epoch)
        self.fgsm.epsilon = current_epsilon
        self.pgd.epsilon = current_epsilon
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.size(0)
            
            # Determine split for clean and adversarial examples
            adv_size = int(batch_size * self.attack_ratio)
            clean_size = batch_size - adv_size
            
            if adv_size > 0:
                # Split batch
                clean_data = data[:clean_size]
                clean_target = target[:clean_size]
                adv_data = data[clean_size:]
                adv_target = target[clean_size:]
                
                # Generate adversarial examples
                if np.random.rand() > 0.5:
                    # Use FGSM
                    adv_examples = self.fgsm.generate(self.model, adv_data, adv_target)
                else:
                    # Use PGD
                    adv_examples = self.pgd.generate(self.model, adv_data, adv_target)
                
                # Combine clean and adversarial examples
                combined_data = torch.cat([clean_data, adv_examples])
                combined_target = torch.cat([clean_target, adv_target])
            else:
                combined_data = data
                combined_target = target
            
            # Forward pass
            self.optimiser.zero_grad()
            outputs = self.model(combined_data)
            loss = criterion(outputs, combined_target)
            
            # Backward pass
            loss.backward()
            self.optimiser.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += combined_target.size(0)
            correct += predicted.eq(combined_target).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'Eps': f'{current_epsilon:.3f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        attack: Optional[str] = None
    ) -> Tuple[float, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in val_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            if attack == 'fgsm':
                # Generate adversarial examples
                data = self.fgsm.generate(self.model, data, target)
            elif attack == 'pgd':
                # Generate adversarial examples
                data = self.pgd.generate(self.model, data, target)
            
            # Evaluate
            with torch.no_grad():
                outputs = self.model(data)
                loss = criterion(outputs, target)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: Optional[str] = None
    ):
        """
        Train model with adversarial training.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_path: Path to save best model
        """
        best_robust_acc = 0
        
        for epoch in range(self.epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Evaluation
            val_loss, val_acc = self.evaluate(val_loader)
            val_robust_loss, val_robust_acc = self.evaluate(val_loader, attack='pgd')
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_robust_acc'].append(val_robust_acc)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Robust Acc (PGD): {val_robust_acc:.2f}%")
            
            # Save best model
            if save_path and val_robust_acc > best_robust_acc:
                best_robust_acc = val_robust_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimiser_state_dict': self.optimiser.state_dict(),
                    'best_robust_acc': best_robust_acc,
                    'history': self.history,
                    'config': self.config
                }, save_path)
                print(f"Saved best model with robust accuracy: {best_robust_acc:.2f}%")


class FreeAdversarialTraining(AdversarialTrainer):
    """Fast adversarial training using free adversarial training method."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device = torch.device('cpu'),
        replay_m: int = 4
    ):
        """
        Initialise free adversarial training.
        
        Args:
            model: Model to train
            config: Configuration dictionary
            device: Device to use
            replay_m: Number of replay iterations
        """
        super().__init__(model, config, device)
        self.replay_m = replay_m
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        criterion: nn.Module = nn.CrossEntropyLoss()
    ) -> Tuple[float, float]:
        """Train for one epoch using free adversarial training."""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        current_epsilon = self.get_epsilon(epoch)
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Initialise perturbation
            delta = torch.zeros_like(data, requires_grad=True)
            
            for _ in range(self.replay_m):
                # Forward pass with perturbation
                outputs = self.model(data + delta)
                loss = criterion(outputs, target)
                
                # Backward pass
                self.optimiser.zero_grad()
                loss.backward()
                
                # Update model
                self.optimiser.step()
                
                # Update perturbation
                grad = delta.grad.detach()
                delta.data = torch.clamp(
                    delta + current_epsilon * grad.sign(),
                    -current_epsilon,
                    current_epsilon
                )
                delta.grad.zero_()
            
            # Statistics (on clean examples for consistency)
            with torch.no_grad():
                outputs = self.model(data)
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                total_loss += loss.item()
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'Eps': f'{current_epsilon:.3f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy


def create_robust_model(
    base_model: nn.Module,
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device = torch.device('cpu'),
    save_path: Optional[str] = None,
    use_free_training: bool = False
) -> nn.Module:
    """
    Create adversarially robust model through adversarial training.
    
    Args:
        base_model: Base model architecture
        config: Configuration dictionary
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to use
        save_path: Path to save trained model
        use_free_training: Whether to use free adversarial training
    
    Returns:
        Trained robust model
    """
    if use_free_training:
        trainer = FreeAdversarialTraining(base_model, config, device)
    else:
        trainer = AdversarialTrainer(base_model, config, device)
    
    trainer.train(train_loader, val_loader, save_path)
    
    return trainer.model