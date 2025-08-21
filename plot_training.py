#!/usr/bin/env python3
"""
Plot training curves from TensorBoard logs for FOMO Task1 training
Creates loss and metrics plots for analysis
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Please install tensorboard: pip install tensorboard")
    sys.exit(1)

def extract_tensorboard_data(log_dir):
    """Extract data from TensorBoard log files"""
    print(f"Loading TensorBoard data from: {log_dir}")
    
    # Find the most recent version directory
    log_path = Path(log_dir)
    version_dirs = [d for d in log_path.iterdir() if d.is_dir() and d.name.startswith('version_')]
    
    if not version_dirs:
        print("No version directories found!")
        return None
    
    # Sort by creation time and get the most recent
    latest_version = sorted(version_dirs, key=lambda x: x.stat().st_mtime)[-1]
    print(f"Using latest version: {latest_version}")
    
    # Load event data
    ea = EventAccumulator(str(latest_version))
    ea.Reload()
    
    # Extract scalar data
    data = {}
    scalar_tags = ea.Tags()['scalars']
    print(f"Available metrics: {scalar_tags}")
    
    for tag in scalar_tags:
        scalar_events = ea.Scalars(tag)
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        data[tag] = {'steps': steps, 'values': values}
    
    return data, latest_version

def plot_training_curves(data, output_dir, title_prefix="FOMO Task1"):
    """Create training curve plots"""
    
    if not data:
        print("No data to plot!")
        return
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up plot style
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    # 1. Loss curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{title_prefix} Training Progress', fontsize=16)
    
    # Training and validation loss
    ax = axes[0, 0]
    for metric_name, metric_data in data.items():
        if 'loss' in metric_name.lower():
            label = metric_name.replace('_', ' ').title()
            ax.plot(metric_data['steps'], metric_data['values'], label=label, linewidth=2)
    
    ax.set_title('Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy metrics
    ax = axes[0, 1]
    for metric_name, metric_data in data.items():
        if any(acc_term in metric_name.lower() for acc_term in ['acc', 'f1', 'auroc']):
            label = metric_name.replace('_', ' ').title()
            ax.plot(metric_data['steps'], metric_data['values'], label=label, linewidth=2)
    
    ax.set_title('Accuracy Metrics')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[1, 0]
    for metric_name, metric_data in data.items():
        if 'lr' in metric_name.lower() or 'learning' in metric_name.lower():
            ax.plot(metric_data['steps'], metric_data['values'], label='Learning Rate', linewidth=2)
            break
    
    ax.set_title('Learning Rate Schedule')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Additional metrics
    ax = axes[1, 1]
    other_metrics = [name for name in data.keys() 
                    if not any(term in name.lower() for term in ['loss', 'acc', 'f1', 'auroc', 'lr', 'learning'])]
    
    for metric_name in other_metrics[:3]:  # Plot up to 3 additional metrics
        metric_data = data[metric_name]
        label = metric_name.replace('_', ' ').title()
        ax.plot(metric_data['steps'], metric_data['values'], label=label, linewidth=2)
    
    ax.set_title('Additional Metrics')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    if other_metrics:
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'training_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {plot_path}")
    
    # Also save as PDF
    pdf_path = output_dir / 'training_curves.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF version saved to: {pdf_path}")
    
    plt.show()
    
    return plot_path

def create_summary_table(data, output_dir):
    """Create a summary table of final metrics"""
    
    output_dir = Path(output_dir)
    
    # Extract final values
    summary = {}
    for metric_name, metric_data in data.items():
        if metric_data['values']:
            final_value = metric_data['values'][-1]
            max_value = max(metric_data['values'])
            min_value = min(metric_data['values'])
            
            summary[metric_name] = {
                'Final': final_value,
                'Max': max_value,
                'Min': min_value,
                'Epochs': len(metric_data['values'])
            }
    
    # Create DataFrame and save
    df = pd.DataFrame(summary).T
    
    # Save as CSV
    csv_path = output_dir / 'training_summary.csv'
    df.to_csv(csv_path)
    print(f"Training summary saved to: {csv_path}")
    
    # Print summary
    print("\n=== Training Summary ===")
    print(df.round(4))
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Plot FOMO Task1 training curves")
    parser.add_argument("--log_dir", 
                       default="/data/weidong/workspace/SSL3D_classification/experiments/debug/classification/fomo_task1",
                       help="Path to TensorBoard log directory")
    parser.add_argument("--output_dir",
                       default="/data/weidong/workspace/SSL3D_classification/experiments/plots",
                       help="Output directory for plots")
    parser.add_argument("--title", 
                       default="FOMO Task1",
                       help="Title prefix for plots")
    
    args = parser.parse_args()
    
    # Extract data from TensorBoard logs
    data, log_version = extract_tensorboard_data(args.log_dir)
    
    if data:
        # Create output directory with version info
        output_dir = Path(args.output_dir) / f"fomo_task1_{log_version.name}"
        
        # Plot training curves
        plot_path = plot_training_curves(data, output_dir, args.title)
        
        # Create summary table
        summary_df = create_summary_table(data, output_dir)
        
        print(f"\n✅ Analysis complete!")
        print(f"📊 Plots saved to: {output_dir}")
        print(f"📈 View training curves: {plot_path}")
        
        # Instructions for TensorBoard
        print(f"\n💡 To view live training in TensorBoard:")
        print(f"   tensorboard --logdir {args.log_dir}")
        print(f"   Then open: http://localhost:6006")
        
    else:
        print("❌ No training data found!")

if __name__ == "__main__":
    main()
