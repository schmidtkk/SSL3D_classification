#!/usr/bin/env python3
"""
Plot training curves from CSV logs (CSVLogger output)
Simple alternative to TensorBoard for FOMO Task1 training visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def find_latest_csv_logs(base_dir):
    """Find the most recent CSV log files"""
    base_path = Path(base_dir)
    
    # Look for CSV files in version directories
    csv_files = list(base_path.rglob("metrics.csv"))
    
    if not csv_files:
        print(f"No metrics.csv files found in {base_dir}")
        return None
        
    # Get the most recent one
    latest_csv = sorted(csv_files, key=lambda x: x.stat().st_mtime)[-1]
    print(f"Using CSV log: {latest_csv}")
    
    return latest_csv

def plot_csv_metrics(csv_file, output_dir):
    """Plot metrics from CSV file"""
    
    # Read CSV data
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows of metrics data")
    print(f"Available columns: {list(df.columns)}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up plotting
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('FOMO Task1 Training Progress', fontsize=16)
    
    # Plot 1: Loss curves
    ax = axes[0, 0]
    loss_cols = [col for col in df.columns if 'loss' in col.lower()]
    for col in loss_cols:
        if not df[col].isna().all():  # Only plot if not all NaN
            ax.plot(df['epoch'], df[col], label=col.replace('_', ' ').title(), linewidth=2)
    
    ax.set_title('Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    if loss_cols:
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy/Performance metrics
    ax = axes[0, 1]
    perf_cols = [col for col in df.columns if any(term in col.lower() 
                 for term in ['acc', 'f1', 'auroc', 'ap', 'balanced'])]
    for col in perf_cols:
        if not df[col].isna().all():
            ax.plot(df['epoch'], df[col], label=col.replace('_', ' ').title(), linewidth=2)
    
    ax.set_title('Performance Metrics')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    if perf_cols:
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Learning rate (if available)
    ax = axes[1, 0]
    lr_cols = [col for col in df.columns if 'lr' in col.lower() or 'learning' in col.lower()]
    for col in lr_cols:
        if not df[col].isna().all():
            ax.plot(df['epoch'], df[col], label='Learning Rate', linewidth=2)
    
    ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    if lr_cols:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Other metrics
    ax = axes[1, 1]
    other_cols = [col for col in df.columns 
                  if col not in ['epoch', 'step'] + loss_cols + perf_cols + lr_cols
                  and not df[col].isna().all()]
    
    for col in other_cols[:3]:  # Plot up to 3 additional metrics
        ax.plot(df['epoch'], df[col], label=col.replace('_', ' ').title(), linewidth=2)
    
    ax.set_title('Additional Metrics')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    if other_cols:
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    plot_path = output_dir / 'training_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {plot_path}")
    
    pdf_path = output_dir / 'training_curves.pdf' 
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF version saved to: {pdf_path}")
    
    plt.show()
    
    # Create summary table
    summary = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['epoch', 'step']]
    
    for col in numeric_cols:
        if not df[col].isna().all():
            final_val = df[col].dropna().iloc[-1] if not df[col].dropna().empty else np.nan
            max_val = df[col].max()
            min_val = df[col].min()
            
            summary[col] = {
                'Final': final_val,
                'Best': max_val if 'loss' not in col.lower() else min_val,
                'Range': f"{min_val:.4f} - {max_val:.4f}"
            }
    
    summary_df = pd.DataFrame(summary).T
    summary_path = output_dir / 'training_summary.csv'
    summary_df.to_csv(summary_path)
    print(f"Training summary saved to: {summary_path}")
    
    print("\n=== Training Summary ===")
    print(summary_df.round(4))
    
    return plot_path

def main():
    parser = argparse.ArgumentParser(description="Plot FOMO Task1 CSV training logs")
    parser.add_argument("--log_dir",
                       default="/data/weidong/workspace/SSL3D_classification/experiments/debug/classification/fomo_task1",
                       help="Path to CSV log directory")
    parser.add_argument("--output_dir", 
                       default="/data/weidong/workspace/SSL3D_classification/experiments/plots",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Find CSV file
    csv_file = find_latest_csv_logs(args.log_dir)
    
    if csv_file:
        # Create plots
        plot_path = plot_csv_metrics(csv_file, args.output_dir)
        print(f"\n✅ CSV analysis complete!")
        print(f"📊 Plots saved to: {args.output_dir}")
    else:
        print("❌ No CSV log files found!")

if __name__ == "__main__":
    main()
