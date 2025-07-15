# Visualisation utilities for analysis and reporting

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from pathlib import Path
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def setup_plotting_style():
    """Set up consistent plotting style."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 12


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    Plot training history curves.
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    
    # Accuracy curves
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    
    # Robust accuracy
    if 'val_robust_acc' in history:
        axes[1, 0].plot(history['val_robust_acc'], label='Robust Accuracy', linewidth=2, color='red')
        axes[1, 0].plot(history['val_acc'], label='Clean Accuracy', linewidth=2, color='blue')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title('Clean vs Robust Accuracy')
        axes[1, 0].legend()
    
    # Learning rate schedule (if available)
    if 'learning_rate' in history:
        axes[1, 1].plot(history['learning_rate'], linewidth=2, color='green')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalise: bool = True
):
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix
        class_names: List of class names
        save_path: Path to save plot
        normalise: Whether to normalise matrix
    """
    if normalise:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='.2f' if normalise else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={'label': 'Proportion' if normalise else 'Count'}
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix' + (' (Normalised)' if normalise else ''))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_bias_analysis(
    bias_report: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Create comprehensive bias analysis visualisation.
    
    Args:
        bias_report: Bias analysis report DataFrame
        save_path: Path to save plot
    """
    # Filter out summary rows
    class_data = bias_report[~bias_report['Class'].isin(['Overall', 'Std Dev'])]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Class Distribution', 'Performance Metrics',
                       'Robustness Comparison', 'Bias Heatmap'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'heatmap'}]]
    )
    
    # 1. Class distribution
    fig.add_trace(
        go.Bar(x=class_data['Class'], y=class_data['Train Proportion'],
               name='Train', marker_color='lightblue'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=class_data['Class'], y=class_data['Test Proportion'],
               name='Test', marker_color='darkblue'),
        row=1, col=1
    )
    
    # 2. Performance metrics
    fig.add_trace(
        go.Bar(x=class_data['Class'], y=class_data['Accuracy'],
               name='Accuracy', marker_color='green'),
        row=1, col=2
    )
    
    # 3. Robustness comparison
    fig.add_trace(
        go.Bar(x=class_data['Class'], y=class_data['FGSM Robustness'],
               name='FGSM', marker_color='orange'),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=class_data['Class'], y=class_data['PGD Robustness'],
               name='PGD', marker_color='red'),
        row=2, col=1
    )
    
    # 4. Bias heatmap
    bias_metrics = class_data[['Accuracy', 'FGSM Robustness', 
                               'PGD Robustness', 'Lighting Sensitivity']].T
    
    fig.add_trace(
        go.Heatmap(
            z=bias_metrics.values,
            x=class_data['Class'].values,
            y=['Accuracy', 'FGSM Rob.', 'PGD Rob.', 'Light Sens.'],
            colorscale='RdYlBu'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Comprehensive Bias Analysis"
    )
    
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()


def plot_robustness_surface(
    epsilon_values: List[float],
    attack_types: List[str],
    accuracy_matrix: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot 3D robustness surface.
    
    Args:
        epsilon_values: List of epsilon values
        attack_types: List of attack types
        accuracy_matrix: Accuracy matrix (attacks x epsilons)
        save_path: Path to save plot
    """
    fig = go.Figure(data=[
        go.Surface(
            x=epsilon_values,
            y=list(range(len(attack_types))),
            z=accuracy_matrix,
            colorscale='Viridis'
        )
    ])
    
    fig.update_layout(
        title='Model Robustness Surface',
        scene=dict(
            xaxis_title='Epsilon',
            yaxis_title='Attack Type',
            zaxis_title='Accuracy (%)',
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(attack_types))),
                ticktext=attack_types
            )
        ),
        width=900,
        height=700
    )
    
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()


def visualise_feature_space(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    method: str = 'tsne',
    save_path: Optional[str] = None
):
    """
    Visualise high-dimensional features in 2D.
    
    Args:
        features: Feature tensor
        labels: Label tensor
        class_names: List of class names
        method: Dimensionality reduction method ('tsne', 'pca', 'umap')
        save_path: Path to save plot
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    # Convert to numpy
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Reduce dimensionality
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        features_2d = reducer.fit_transform(features_np)
    elif method == 'pca':
        reducer = PCA(n_components=2)
        features_2d = reducer.fit_transform(features_np)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    for i in range(len(class_names)):
        mask = labels_np == i
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            label=class_names[i],
            alpha=0.6,
            s=50
        )
    
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(f'Feature Space Visualisation ({method.upper()})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_attack_comparison(
    attack_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
):
    """
    Plot comparison of different attacks.
    
    Args:
        attack_results: Results for different attacks
        save_path: Path to save plot
    """
    attacks = list(attack_results.keys())
    metrics = ['accuracy', 'success_rate', 'l2_norm', 'linf_norm']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [attack_results[attack].get(metric, 0) for attack in attacks]
        
        axes[i].bar(attacks, values, color=plt.cm.viridis(np.linspace(0, 1, len(attacks))))
        axes[i].set_xlabel('Attack Type')
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].set_title(f'{metric.replace("_", " ").title()} by Attack Type')
        
        # Add value labels on bars
        for j, v in enumerate(values):
            axes[i].text(j, v + max(values) * 0.01, f'{v:.2f}', 
                        ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_model_report(
    config: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    bias_report: pd.DataFrame,
    save_dir: str
):
    """
    Create comprehensive model evaluation report.
    
    Args:
        config: Configuration dictionary
        evaluation_results: Evaluation results
        bias_report: Bias analysis report
        save_dir: Directory to save report
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create HTML report
    html_content = f"""
    <html>
    <head>
        <title>Model Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; color: #0066cc; }}
        </style>
    </head>
    <body>
        <h1>Adversarially Robust Model Evaluation Report</h1>
        
        <h2>Model Configuration</h2>
        <ul>
            <li>Architecture: {config['model']['architecture']}</li>
            <li>Dataset: {config['data']['dataset_name']}</li>
            <li>Training Epochs: {config['training']['epochs']}</li>
            <li>Adversarial Training: {config['adversarial']['training']['enable']}</li>
        </ul>
        
        <h2>Performance Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Clean Accuracy</td>
                <td class="metric">{evaluation_results.get('clean_accuracy', 0):.2f}%</td>
            </tr>
            <tr>
                <td>FGSM Robustness</td>
                <td class="metric">{evaluation_results.get('fgsm_accuracy', 0):.2f}%</td>
            </tr>
            <tr>
                <td>PGD Robustness</td>
                <td class="metric">{evaluation_results.get('pgd_accuracy', 0):.2f}%</td>
            </tr>
        </table>
        
        <h2>Bias Analysis</h2>
        {bias_report.to_html()}
        
        <h2>Visualisations</h2>
        <p>See accompanying plots in the results directory.</p>
        
        <hr>
        <p><i>Report generated automatically by the adversarial robustness pipeline.</i></p>
    </body>
    </html>
    """
    
    report_path = save_dir / "evaluation_report.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"Report saved to: {report_path}")