# Training utilities for model training and validation

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm

from .metrics import MetricsTracker


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min'
    ):
        """
        Initialise early stopping.
        
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
    
    def __call__(self, value: float) -> bool:
        """Check if should stop training."""
        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class LearningRateScheduler:
    """Custom learning rate scheduler."""
    
    def __init__(
        self,
        optimiser: optim.Optimizer,
        config: Dict[str, Any]
    ):
        """
        Initialise learning rate scheduler.
        
        Args:
            optimiser: Optimiser instance
            config: Scheduler configuration
        """
        self.optimiser = optimiser
        self.config = config
        self.scheduler_type = config['type']
        
        if self.scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimiser,
                T_max=config.get('T_max', 100)
            )
        elif self.scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                optimiser,
                step_size=config['step_size'],
                gamma=config['gamma']
            )
        elif self.scheduler_type == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                optimiser,
                gamma=config['gamma']
            )
        elif self.scheduler_type == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimiser,
                mode='min',
                patience=config.get('patience', 5),
                factor=config.get('factor', 0.5)
            )
        else:
            self.scheduler = None
    
    def step(self, metrics: Optional[float] = None):
        """Step the scheduler."""
        if self.scheduler is not None:
            if self.scheduler_type == 'reduce_on_plateau':
                self.scheduler.step(metrics)
            else:
                self.scheduler.step()
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimiser.param_groups[0]['lr']


class TrainingLogger:
    """Logger for training progress and metrics."""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        config: Dict[str, Any]
    ):
        """
        Initialise training logger.
        
        Args:
            log_dir: Directory for logs
            experiment_name: Name of experiment
            config: Configuration dictionary
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.config = config
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialise TensorBoard writer
        self.writer = SummaryWriter(self.log_dir / experiment_name)
        
        # Save configuration
        config_path = self.log_dir / f"{experiment_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Initialise history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'epoch_time': []
        }
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float,
        epoch_time: float
    ):
        """Log epoch metrics."""
        # Update history
        self.history['train_loss'].append(train_metrics.get('loss', 0))
        self.history['train_acc'].append(train_metrics.get('accuracy', 0))
        self.history['val_loss'].append(val_metrics.get('loss', 0))
        self.history['val_acc'].append(val_metrics.get('accuracy', 0))
        self.history['learning_rate'].append(learning_rate)
        self.history['epoch_time'].append(epoch_time)
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/train', train_metrics.get('loss', 0), epoch)
        self.writer.add_scalar('Loss/val', val_metrics.get('loss', 0), epoch)
        self.writer.add_scalar('Accuracy/train', train_metrics.get('accuracy', 0), epoch)
        self.writer.add_scalar('Accuracy/val', val_metrics.get('accuracy', 0), epoch)
        self.writer.add_scalar('Learning_Rate', learning_rate, epoch)
        
        # Log additional metrics if available (only scalar values)
        for key, value in train_metrics.items():
            if key not in ['loss', 'accuracy', 'confusion_matrix', 'per_class_metrics']:
                # Check if value is scalar
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        for key, value in val_metrics.items():
            if key not in ['loss', 'accuracy', 'confusion_matrix', 'per_class_metrics']:
                # Check if value is scalar
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Val/{key}', value, epoch)
    
    def log_images(
        self,
        epoch: int,
        images: torch.Tensor,
        title: str = 'samples'
    ):
        """Log image samples."""
        self.writer.add_images(title, images, epoch)
    
    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimiser: optim.Optimizer,
        best_metric: float,
        is_best: bool = False
    ):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            'best_metric': best_metric,
            'history': self.history,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.log_dir / f"{self.experiment_name}_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.log_dir / f"{self.experiment_name}_best.pth"
            torch.save(checkpoint, best_path)
    
    def close(self):
        """Close logger."""
        self.writer.close()
        
        # Save final history
        history_path = self.log_dir / f"{self.experiment_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimiser: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    augmentation_fn: Optional[Callable] = None
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimiser: Optimiser
        device: Device to use
        epoch: Current epoch
        config: Configuration dictionary
        augmentation_fn: Optional augmentation function
    
    Returns:
        Training metrics
    """
    model.train()
    tracker = MetricsTracker(config['model']['num_classes'])
    total_loss = 0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        # Apply augmentation if provided
        if augmentation_fn is not None:
            data, target, mixed_targets, lam = augmentation_fn(data, target)
        else:
            mixed_targets = None
            lam = None
        
        # Forward pass
        optimiser.zero_grad()
        outputs = model(data)
        
        # Calculate loss
        if mixed_targets is not None:
            # Mixup/CutMix loss
            loss = lam * criterion(outputs, mixed_targets[0]) + (1 - lam) * criterion(outputs, mixed_targets[1])
        else:
            loss = criterion(outputs, target)
        
        # Add regularisation if configured
        if 'weight_decay' in config['training'] and config['training']['weight_decay'] > 0:
            l2_reg = sum(p.pow(2).sum() for p in model.parameters())
            loss += config['training']['weight_decay'] * l2_reg
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping if configured
        if 'gradient_clip' in config['training']:
            nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        
        optimiser.step()
        
        # Update metrics
        with torch.no_grad():
            predictions = outputs.argmax(dim=1)
            probabilities = torch.softmax(outputs, dim=1)
            
            # For mixed augmentation, use original targets for metrics
            if mixed_targets is not None:
                tracker.update(predictions, target, probabilities, loss.item())
            else:
                tracker.update(predictions, target, probabilities, loss.item())
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    # Compute epoch metrics
    metrics = tracker.compute_metrics()
    metrics['loss'] = total_loss / num_batches
    
    return metrics


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Validate model.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        config: Configuration dictionary
    
    Returns:
        Validation metrics
    """
    model.eval()
    tracker = MetricsTracker(config['model']['num_classes'])
    total_loss = 0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validation'):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # Update metrics
            predictions = outputs.argmax(dim=1)
            probabilities = torch.softmax(outputs, dim=1)
            tracker.update(predictions, target, probabilities, loss.item())
            
            total_loss += loss.item()
    
    # Compute metrics
    metrics = tracker.compute_metrics()
    metrics['loss'] = total_loss / num_batches
    
    return metrics


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    experiment_name: str = 'experiment',
    augmentation_fn: Optional[Callable] = None
) -> nn.Module:
    """
    Complete training pipeline.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        device: Device to use
        experiment_name: Name for experiment
        augmentation_fn: Optional augmentation function
    
    Returns:
        Trained model
    """
    # Setup
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialise helpers
    logger = TrainingLogger(config['logging']['save_dir'], experiment_name, config)
    scheduler = LearningRateScheduler(optimiser, config['training']['scheduler'])
    
    early_stopping = None
    if config['training']['early_stopping']['enable']:
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta'],
            mode='max'  # Monitoring validation accuracy
        )
    
    best_val_acc = 0
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        start_time = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimiser,
            device, epoch, config, augmentation_fn
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, config)
        
        # Update scheduler
        current_lr = scheduler.get_lr()
        scheduler.step(val_metrics['loss'])
        
        epoch_time = time.time() - start_time
        
        # Log metrics
        logger.log_epoch(epoch, train_metrics, val_metrics, current_lr, epoch_time)
        
        # Save checkpoint
        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']
        
        logger.save_checkpoint(epoch, model, optimiser, best_val_acc, is_best)
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}, Time: {epoch_time:.2f}s")
        
        # Early stopping
        if early_stopping is not None:
            if early_stopping(val_metrics['accuracy']):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
    
    logger.close()
    return model