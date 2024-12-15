#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def load_optimization_report(file_path: str = 'hyperparameter_optimization_report.json') -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)

def create_trial_dataframe(report: dict) -> pd.DataFrame:
    trials = []
    for trial in report['optimization_history']:
        trial_data = {
            'trial': trial['trial'],
            'accuracy': trial['accuracy'],
        }
        trial_data.update(trial['params'])
        trials.append(trial_data)
    return pd.DataFrame(trials)

def plot_accuracy_history(df: pd.DataFrame, save_dir: Path):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='trial', y='accuracy', alpha=0.6, label='Trial Accuracy')
    sns.lineplot(data=df, x='trial', y='accuracy', color='blue', alpha=0.3)
    plt.plot(df['trial'], df['accuracy'].cummax(), 'r--', label='Best Accuracy')
    plt.xlabel('Trial Number')
    plt.ylabel('Accuracy (%)')
    plt.title('Optimization Progress', pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'accuracy_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_parameter_heatmap(df: pd.DataFrame, save_dir: Path):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Create mask for upper triangle
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f',
                mask=mask, square=True, linewidths=0.5)
    plt.title('Parameter Correlation Heatmap', pad=20)
    plt.tight_layout()
    plt.savefig(save_dir / 'parameter_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_parameter_importance(report: dict, save_dir: Path):
    if report['importance'] is None:
        return

    importance = report['importance']
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame(
        list(importance.items()),
        columns=['Parameter', 'Importance']
    ).sort_values('Importance', ascending=True)

    sns.barplot(data=importance_df, x='Importance', y='Parameter')
    plt.xlabel('Importance Score')
    plt.title('Parameter Importance Analysis', pad=20)
    plt.tight_layout()
    plt.savefig(save_dir / 'parameter_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_layer_size_distribution(df: pd.DataFrame, save_dir: Path):
    layer_cols = [col for col in df.columns if col.startswith('hidden_size_')]
    if not layer_cols:
        return

    plt.figure(figsize=(12, 6))
    for col in layer_cols:
        sns.kdeplot(data=df[col], label=f'Layer {col[-1]}', fill=True, alpha=0.3)
    plt.xlabel('Layer Size (neurons)')
    plt.ylabel('Density')
    plt.title('Distribution of Layer Sizes Across Trials', pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'layer_size_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_best_architectures(df: pd.DataFrame, save_dir: Path, top_n: int = 5):
    best_trials = df.nlargest(top_n, 'accuracy')
    layer_cols = [col for col in df.columns if col.startswith('hidden_size_')]
    if not layer_cols:
        return

    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("husl", n_colors=top_n)

    for (idx, row), color in zip(best_trials.iterrows(), colors):
        layer_sizes = [row[col] for col in layer_cols if not pd.isna(row[col])]
        plt.plot(range(len(layer_sizes)), layer_sizes, 'o-',
                color=color, linewidth=2, markersize=8,
                label=f'Trial {int(row["trial"])} ({row["accuracy"]:.2f}%)')

    plt.xlabel('Layer Index')
    plt.ylabel('Layer Size (neurons)')
    plt.title('Top Performing Network Architectures', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'best_architectures.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_parameter_violin_plots(df: pd.DataFrame, save_dir: Path):
    params = [col for col in df.columns if col not in ['trial', 'accuracy']]
    
    for param in params:
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, x=param, y='accuracy', orient='h')
        plt.title(f'Accuracy Distribution vs {param}', pad=20)
        plt.tight_layout()
        plt.savefig(save_dir / f'{param}_violin.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    plt.style.use('tableau-colorblind10')

    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.2)

    save_dir = Path('optimization_plots')
    save_dir.mkdir(exist_ok=True)

    report = load_optimization_report()
    df = create_trial_dataframe(report)

    # Create all plots
    print("📊 Creating visualization plots...")

    print("  • Plotting accuracy history")
    plot_accuracy_history(df, save_dir)

    print("  • Creating parameter correlation heatmap")
    plot_parameter_heatmap(df, save_dir)

    print("  • Plotting parameter importance")
    plot_parameter_importance(report, save_dir)

    print("  • Plotting layer size distribution")
    plot_layer_size_distribution(df, save_dir)

    print("  • Visualizing best architectures")
    plot_best_architectures(df, save_dir)

    print("  • Creating parameter violin plots")
    create_parameter_violin_plots(df, save_dir)

    print(f"\n✨ All plots have been saved to the '{save_dir}' directory")
    print("\nGenerated plots:")
    for plot_file in save_dir.glob('*.png'):
        print(f"  • {plot_file.name}")

if __name__ == "__main__":
    main()