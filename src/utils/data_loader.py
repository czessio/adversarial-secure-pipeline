# Data loading utilities for image classification pipeline

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple, Optional, Dict, Any
import yaml
from pathlib import Path


class CustomDataset(Dataset):
    """Custom dataset wrapper for applying preprocessing transforms."""
    
    def __init__(self, base_dataset: Dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_data_transforms(config: Dict[str, Any], train: bool = True) -> transforms.Compose:
    """Create data transformation pipeline based on configuration."""
    transform_list = []
    
    if train and config['preprocessing']['augmentation']['enable']:
        aug_config = config['preprocessing']['augmentation']
        
        if aug_config['random_crop']:
            transform_list.append(transforms.RandomCrop(32, padding=4))
        
        if aug_config['random_horizontal_flip']:
            transform_list.append(transforms.RandomHorizontalFlip())
        
        if any(aug_config['colour_jitter'].values()):
            transform_list.append(transforms.ColorJitter(
                brightness=aug_config['colour_jitter']['brightness'],
                contrast=aug_config['colour_jitter']['contrast'],
                saturation=aug_config['colour_jitter']['saturation'],
                hue=aug_config['colour_jitter']['hue']
            ))
    
    # Always apply these transforms
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(transform_list)


def get_dataset(config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
    """Load dataset based on configuration."""
    dataset_name = config['data']['dataset_name']
    data_dir = Path(config['data']['data_dir'])
    
    if dataset_name == "CIFAR10":
        dataset_class = datasets.CIFAR10
        config['model']['num_classes'] = 10
    elif dataset_name == "CIFAR100":
        dataset_class = datasets.CIFAR100
        config['model']['num_classes'] = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Load full training dataset
    full_train_dataset = dataset_class(
        root=data_dir / 'raw',
        train=True,
        download=True,
        transform=None  # We'll apply transforms later
    )
    
    # Split into train and validation
    val_split = config['data']['validation_split']
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['data']['random_seed'])
    )
    
    # Load test dataset
    test_dataset = dataset_class(
        root=data_dir / 'raw',
        train=False,
        download=True,
        transform=None
    )
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(
    config: Dict[str, Any],
    train_transform: Optional[transforms.Compose] = None,
    val_transform: Optional[transforms.Compose] = None,
    test_transform: Optional[transforms.Compose] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing."""
    
    # Get datasets
    train_dataset, val_dataset, test_dataset = get_dataset(config)
    
    # Apply transforms
    if train_transform is None:
        train_transform = get_data_transforms(config, train=True)
    if val_transform is None:
        val_transform = get_data_transforms(config, train=False)
    if test_transform is None:
        test_transform = get_data_transforms(config, train=False)
    
    # Wrap datasets with transforms
    train_dataset = CustomDataset(train_dataset, train_transform)
    val_dataset = CustomDataset(val_dataset, val_transform)
    test_dataset = CustomDataset(test_dataset, test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True if config['hardware']['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True if config['hardware']['device'] == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True if config['hardware']['device'] == 'cuda' else False
    )
    
    return train_loader, val_loader, test_loader


def get_class_names(dataset_name: str) -> list:
    """Get class names for the dataset."""
    if dataset_name == "CIFAR10":
        return ['plane', 'car', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset_name == "CIFAR100":
        # Return CIFAR-100 fine labels (abbreviated for space)
        return [f"class_{i}" for i in range(100)]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")