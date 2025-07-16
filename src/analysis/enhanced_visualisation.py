# Enhanced visualisation utilities for comprehensive reporting
# src/analysis/enhanced_visualisation.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime


def setup_plotting_style():
    """Set up consistent plotting style."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 12


def create_model_report(
    config: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    bias_report: pd.DataFrame,
    save_dir: str
):
    """
    Create basic model evaluation report.
    
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
        {bias_report.to_html() if bias_report is not None else '<p>No bias analysis available</p>'}
        
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


class ComprehensiveVisualiser:
    """Enhanced visualiser for generating all figures and reports."""
    
    def __init__(self, results_dir: str, experiment_name: str):
        """
        Initialise visualiser.
        
        Args:
            results_dir: Directory to save results
            experiment_name: Name of experiment
        """
        self.results_dir = Path(results_dir)
        self.experiment_name = experiment_name
        self.figures_dir = self.results_dir / 'figures'
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
    
    def generate_all_visualisations(
        self,
        evaluation_results: Dict[str, pd.DataFrame],
        bias_report: pd.DataFrame,
        training_history: Dict[str, List[float]],
        config: Dict[str, Any]
    ):
        """Generate all visualisations for the experiment."""
        print("\nGenerating comprehensive visualisations...")
        
        # 1. Training visualisations
        self.plot_training_curves(training_history)
        
        # 2. Robustness visualisations
        if evaluation_results:
            self.plot_robustness_comparison(evaluation_results)
            self.plot_attack_effectiveness(evaluation_results)
        
        # 3. Bias visualisations
        if bias_report is not None:
            self.plot_class_performance_analysis(bias_report)
            self.plot_bias_heatmap(bias_report)
        
        # 4. Model comparison
        if len(evaluation_results) > 1:
            self.plot_model_comparison(evaluation_results)
        
        # 5. Generate summary dashboard
        self.create_summary_dashboard(evaluation_results, bias_report, config)
        
        print(f"All visualisations saved to: {self.figures_dir}")
    
    def plot_training_curves(self, history: Dict[str, List[float]]):
        """Plot comprehensive training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss curves
        axes[0, 0].plot(history.get('train_loss', []), label='Train Loss', linewidth=2)
        axes[0, 0].plot(history.get('val_loss', []), label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(history.get('train_acc', []), label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(history.get('val_acc', []), label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Robust accuracy if available
        if 'val_robust_acc' in history and history['val_robust_acc']:
            axes[1, 0].plot(history['val_robust_acc'], label='Robust Accuracy', linewidth=2, color='red')
            axes[1, 0].plot(history.get('val_acc', []), label='Clean Accuracy', linewidth=2, color='blue')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy (%)')
            axes[1, 0].set_title('Clean vs Robust Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        if 'learning_rate' in history and history['learning_rate']:
            axes[1, 1].plot(history['learning_rate'], linewidth=2, color='green')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_robustness_comparison(self, evaluation_results: Dict[str, pd.DataFrame]):
        """Plot robustness comparison across models and attacks."""
        fig = plt.figure(figsize=(15, 10))
        
        # Create subplot for each model
        num_models = len(evaluation_results)
        for i, (model_name, results_df) in enumerate(evaluation_results.items()):
            ax = plt.subplot(num_models, 1, i + 1)
            
            # Group by attack type
            for attack in results_df['attack'].unique():
                if attack != 'None':
                    attack_data = results_df[results_df['attack'] == attack]
                    ax.plot(attack_data['epsilon'], attack_data['accuracy'], 
                           marker='o', linewidth=2, markersize=8, label=attack)
            
            # Add clean accuracy line
            clean_acc = results_df[results_df['attack'] == 'None']['accuracy'].iloc[0]
            ax.axhline(y=clean_acc, color='green', linestyle='--', label='Clean Accuracy')
            
            ax.set_xlabel('Epsilon (ε)')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'{model_name} - Robustness Against Adversarial Attacks')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'robustness_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_attack_effectiveness(self, evaluation_results: Dict[str, pd.DataFrame]):
        """Plot attack effectiveness analysis."""
        # Combine results for standard epsilon values
        standard_eps = 0.03
        
        attack_effectiveness = []
        for model_name, results_df in evaluation_results.items():
            for attack in results_df['attack'].unique():
                if attack != 'None':
                    attack_data = results_df[
                        (results_df['attack'] == attack) & 
                        (results_df['epsilon'] == standard_eps)
                    ]
                    if not attack_data.empty:
                        clean_acc = results_df[results_df['attack'] == 'None']['accuracy'].iloc[0]
                        robust_acc = attack_data['accuracy'].iloc[0]
                        attack_effectiveness.append({
                            'Model': model_name,
                            'Attack': attack,
                            'Clean Accuracy': clean_acc,
                            'Robust Accuracy': robust_acc,
                            'Accuracy Drop': clean_acc - robust_acc
                        })
        
        if attack_effectiveness:
            eff_df = pd.DataFrame(attack_effectiveness)
            
            # Create grouped bar chart
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = np.arange(len(eff_df['Attack'].unique()))
            width = 0.35 / len(evaluation_results)
            
            for i, model in enumerate(eff_df['Model'].unique()):
                model_data = eff_df[eff_df['Model'] == model]
                offset = (i - len(evaluation_results)/2) * width
                ax.bar(x + offset, model_data['Accuracy Drop'], width, label=model)
            
            ax.set_xlabel('Attack Type')
            ax.set_ylabel('Accuracy Drop (%)')
            ax.set_title(f'Attack Effectiveness at ε={standard_eps}')
            ax.set_xticks(x)
            ax.set_xticklabels(eff_df['Attack'].unique())
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'attack_effectiveness.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_class_performance_analysis(self, bias_report: pd.DataFrame):
        """Plot detailed class performance analysis."""
        # Filter out summary rows
        class_data = bias_report[~bias_report['Class'].isin(['Overall', 'Std Dev'])]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Accuracy by class
        axes[0, 0].bar(class_data['Class'], class_data['Accuracy'], color='skyblue')
        axes[0, 0].axhline(y=class_data['Accuracy'].mean(), color='red', linestyle='--', label='Mean')
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Per-Class Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].legend()
        
        # 2. Robustness comparison
        x = np.arange(len(class_data))
        width = 0.35
        axes[0, 1].bar(x - width/2, class_data['FGSM Robustness'], width, label='FGSM', color='orange')
        axes[0, 1].bar(x + width/2, class_data['PGD Robustness'], width, label='PGD', color='red')
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Robustness (%)')
        axes[0, 1].set_title('Per-Class Robustness Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(class_data['Class'], rotation=45)
        axes[0, 1].legend()
        
        # 3. Class distribution
        axes[1, 0].bar(class_data['Class'], class_data['Train Proportion'], alpha=0.7, label='Train')
        axes[1, 0].bar(class_data['Class'], class_data['Test Proportion'], alpha=0.7, label='Test')
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Proportion')
        axes[1, 0].set_title('Class Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend()
        
        # 4. Performance vs Distribution
        axes[1, 1].scatter(class_data['Train Proportion'], class_data['Accuracy'], s=100, alpha=0.6)
        for i, txt in enumerate(class_data['Class']):
            axes[1, 1].annotate(txt, (class_data['Train Proportion'].iloc[i], 
                                     class_data['Accuracy'].iloc[i]))
        axes[1, 1].set_xlabel('Training Proportion')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Accuracy vs Training Distribution')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'class_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_bias_heatmap(self, bias_report: pd.DataFrame):
        """Create bias analysis heatmap."""
        # Filter out summary rows
        class_data = bias_report[~bias_report['Class'].isin(['Overall', 'Std Dev'])]
        
        # Prepare data for heatmap
        metrics = ['Accuracy', 'FGSM Robustness', 'PGD Robustness', 'Lighting Sensitivity']
        heatmap_data = class_data[metrics].T
        heatmap_data.columns = class_data['Class'].values
        
        # Normalise each metric to [0, 1] for better visualisation
        heatmap_normalised = (heatmap_data - heatmap_data.min(axis=1).values.reshape(-1, 1)) / \
                            (heatmap_data.max(axis=1).values.reshape(-1, 1) - 
                             heatmap_data.min(axis=1).values.reshape(-1, 1))
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_normalised, annot=True, fmt='.2f', cmap='RdYlBu', 
                   cbar_kws={'label': 'Normalised Score'})
        plt.title('Bias Analysis Heatmap (Normalised)')
        plt.xlabel('Class')
        plt.ylabel('Metric')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'bias_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_comparison(self, evaluation_results: Dict[str, pd.DataFrame]):
        """Create model comparison visualisation."""
        # Extract key metrics for comparison
        comparison_data = []
        
        for model_name, results_df in evaluation_results.items():
            clean_acc = results_df[results_df['attack'] == 'None']['accuracy'].iloc[0]
            
            # Get robust accuracy at standard epsilon
            fgsm_data = results_df[
                (results_df['attack'] == 'FGSMAttack') & 
                (results_df['epsilon'] == 0.03)
            ]
            pgd_data = results_df[
                (results_df['attack'] == 'PGDAttack') & 
                (results_df['epsilon'] == 0.03)
            ]
            
            comparison_data.append({
                'Model': model_name,
                'Clean Accuracy': clean_acc,
                'FGSM Robustness': fgsm_data['accuracy'].iloc[0] if not fgsm_data.empty else 0,
                'PGD Robustness': pgd_data['accuracy'].iloc[0] if not pgd_data.empty else 0
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Create radar chart
        categories = ['Clean Accuracy', 'FGSM Robustness', 'PGD Robustness']
        
        fig = go.Figure()
        
        for _, row in comp_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[cat] for cat in categories],
                theta=categories,
                fill='toself',
                name=row['Model']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Model Performance Comparison"
        )
        
        fig.write_html(self.figures_dir / 'model_comparison_radar.html')
    
    def create_summary_dashboard(
        self,
        evaluation_results: Dict[str, pd.DataFrame],
        bias_report: pd.DataFrame,
        config: Dict[str, Any]
    ):
        """Create interactive summary dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance Overview', 'Attack Robustness',
                           'Class-wise Performance', 'Training Configuration'),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'table'}]]
        )
        
        # 1. Model performance overview
        if evaluation_results:
            model_names = []
            clean_accs = []
            robust_accs = []
            
            for model_name, results_df in evaluation_results.items():
                model_names.append(model_name)
                clean_accs.append(results_df[results_df['attack'] == 'None']['accuracy'].iloc[0])
                
                pgd_data = results_df[
                    (results_df['attack'] == 'PGDAttack') & 
                    (results_df['epsilon'] == 0.03)
                ]
                robust_accs.append(pgd_data['accuracy'].iloc[0] if not pgd_data.empty else 0)
            
            fig.add_trace(
                go.Bar(name='Clean', x=model_names, y=clean_accs),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(name='Robust', x=model_names, y=robust_accs),
                row=1, col=1
            )
        
        # 2. Attack robustness curves
        if evaluation_results:
            for model_name, results_df in evaluation_results.items():
                for attack in ['FGSMAttack', 'PGDAttack']:
                    attack_data = results_df[results_df['attack'] == attack]
                    if not attack_data.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=attack_data['epsilon'],
                                y=attack_data['accuracy'],
                                mode='lines+markers',
                                name=f'{model_name}-{attack}'
                            ),
                            row=1, col=2
                        )
        
        # 3. Class-wise performance
        if bias_report is not None:
            class_data = bias_report[~bias_report['Class'].isin(['Overall', 'Std Dev'])]
            fig.add_trace(
                go.Bar(x=class_data['Class'], y=class_data['Accuracy']),
                row=2, col=1
            )
        
        # 4. Configuration table
        config_data = [
            ['Dataset', config['data']['dataset_name']],
            ['Architecture', config['model']['architecture']],
            ['Epochs', str(config['training']['epochs'])],
            ['Batch Size', str(config['data']['batch_size'])],
            ['Learning Rate', str(config['training']['learning_rate'])],
            ['Adversarial Training', 'Yes' if config['adversarial']['training']['enable'] else 'No']
        ]
        
        fig.add_trace(
            go.Table(
                cells=dict(values=list(zip(*config_data)))
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            showlegend=True,
            title_text=f"Experiment Summary - {self.experiment_name}"
        )
        
        # Save dashboard
        fig.write_html(self.figures_dir / 'summary_dashboard.html')
        
        # Also create a timestamp file
        timestamp_file = self.figures_dir / 'experiment_timestamp.txt'
        with open(timestamp_file, 'w') as f:
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: {config['data']['dataset_name']} - {config['model']['architecture']}\n")


def generate_final_report(
    experiment_dir: Path,
    config: Dict[str, Any],
    evaluation_results: Dict[str, pd.DataFrame],
    bias_report: pd.DataFrame,
    training_history: Dict[str, List[float]]
):
    """Generate comprehensive final report with all visualisations."""
    experiment_name = experiment_dir.name
    results_dir = experiment_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualiser
    visualiser = ComprehensiveVisualiser(str(results_dir), experiment_name)
    
    # Generate all visualisations
    visualiser.generate_all_visualisations(
        evaluation_results,
        bias_report,
        training_history,
        config
    )
    
    # Create final HTML report
    create_final_html_report(
        results_dir,
        experiment_name,
        config,
        evaluation_results,
        bias_report
    )


def create_final_html_report(
    results_dir: Path,
    experiment_name: str,
    config: Dict[str, Any],
    evaluation_results: Dict[str, pd.DataFrame],
    bias_report: pd.DataFrame
):
    """Create final HTML report with embedded visualisations."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Adversarial Robustness Experiment Report - {experiment_name}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: auto;
                background-color: white;
                padding: 30px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #0066cc;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #666;
                margin-top: 30px;
            }}
            .metric-box {{
                display: inline-block;
                padding: 20px;
                margin: 10px;
                background-color: #e8f2ff;
                border-radius: 8px;
                text-align: center;
            }}
            .metric-value {{
                font-size: 36px;
                font-weight: bold;
                color: #0066cc;
            }}
            .metric-label {{
                font-size: 14px;
                color: #666;
                margin-top: 5px;
            }}
            img {{
                max-width: 100%;
                height: auto;
                margin: 20px 0;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            .timestamp {{
                color: #888;
                font-size: 12px;
                text-align: right;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Adversarial Robustness Experiment Report</h1>
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Experiment Configuration</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Experiment Name</td><td>{experiment_name}</td></tr>
                <tr><td>Dataset</td><td>{config['data']['dataset_name']}</td></tr>
                <tr><td>Model Architecture</td><td>{config['model']['architecture']}</td></tr>
                <tr><td>Training Epochs</td><td>{config['training']['epochs']}</td></tr>
                <tr><td>Batch Size</td><td>{config['data']['batch_size']}</td></tr>
                <tr><td>Learning Rate</td><td>{config['training']['learning_rate']}</td></tr>
                <tr><td>Adversarial Training</td><td>{'Enabled' if config['adversarial']['training']['enable'] else 'Disabled'}</td></tr>
                <tr><td>Quantisation</td><td>{'Enabled' if config['quantization']['enable'] else 'Disabled'}</td></tr>
            </table>
    """
    
    # Add performance summary
    if evaluation_results:
        html_content += """
            <h2>Performance Summary</h2>
            <div style="text-align: center;">
        """
        
        for model_name, results_df in evaluation_results.items():
            clean_acc = results_df[results_df['attack'] == 'None']['accuracy'].iloc[0]
            pgd_data = results_df[(results_df['attack'] == 'PGDAttack') & (results_df['epsilon'] == 0.03)]
            pgd_acc = pgd_data['accuracy'].iloc[0] if not pgd_data.empty else 0
            
            html_content += f"""
                <div class="metric-box">
                    <div class="metric-value">{clean_acc:.1f}%</div>
                    <div class="metric-label">{model_name} Clean Accuracy</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{pgd_acc:.1f}%</div>
                    <div class="metric-label">{model_name} Robust Accuracy (PGD)</div>
                </div>
            """
        
        html_content += "</div>"
    
    # Add visualisations
    html_content += """
        <h2>Training Progress</h2>
        <img src="figures/training_curves.png" alt="Training Curves">
        
        <h2>Robustness Analysis</h2>
        <img src="figures/robustness_comparison.png" alt="Robustness Comparison">
        <img src="figures/attack_effectiveness.png" alt="Attack Effectiveness">
        
        <h2>Bias Analysis</h2>
        <img src="figures/class_performance_analysis.png" alt="Class Performance Analysis">
        <img src="figures/bias_heatmap.png" alt="Bias Heatmap">
        
        <h2>Interactive Visualisations</h2>
        <p><a href="figures/model_comparison_radar.html">View Model Comparison Radar Chart</a></p>
        <p><a href="figures/summary_dashboard.html">View Interactive Summary Dashboard</a></p>
    """
    
    # Add bias patterns if available
    if bias_report is not None:
        html_content += """
            <h2>Identified Bias Patterns</h2>
            <p>The following bias patterns were identified in the model:</p>
            <ul>
        """
        # This would need the bias patterns from the analyser
        html_content += """
                <li>Check bias_analysis.csv for detailed patterns</li>
            </ul>
        """
    
    html_content += """
            <h2>Additional Files</h2>
            <ul>
                <li><a href="bias_analysis.csv">Detailed Bias Analysis (CSV)</a></li>
                <li><a href="evaluation_results.json">Complete Evaluation Results (JSON)</a></li>
                <li><a href="figures/experiment_timestamp.txt">Experiment Timestamp</a></li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Save report
    report_path = results_dir / 'final_report.html'
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"\nFinal report saved to: {report_path}")