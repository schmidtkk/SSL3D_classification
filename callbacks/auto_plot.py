"""
Automatic CSV Log Plotting Callback
Automatically generates training curve plots every N epochs during training.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import Callback
import logging

logger = logging.getLogger(__name__)


class AutoPlotCallback(Callback):
    """
    Callback that automatically generates training plots every N epochs.
    
    Args:
        plot_every_n_epochs (int): Generate plots every N epochs (default: 10)
        save_dir (str): Directory to save plots (default: experiments/plots)
        plot_format (str): Format for saved plots ('png', 'pdf', 'both') (default: 'png')
        show_plots (bool): Whether to display plots (default: False for headless training)
    """
    
    def __init__(
        self, 
        plot_every_n_epochs: int = 10, 
        save_dir: str = "experiments/plots",
        plot_format: str = "png",
        show_plots: bool = False
    ):
        super().__init__()
        self.plot_every_n_epochs = plot_every_n_epochs
        self.save_dir = Path(save_dir)
        self.plot_format = plot_format
        self.show_plots = show_plots
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib backend for headless operation
        if not show_plots:
            plt.switch_backend('Agg')
    
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Called when the validation epoch ends."""
        current_epoch = trainer.current_epoch
        
        # Check if we should plot this epoch
        if (current_epoch + 1) % self.plot_every_n_epochs == 0:
            self._generate_plots(trainer, current_epoch)
    
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Called when training ends - always generate final plots."""
        self._generate_plots(trainer, trainer.current_epoch, is_final=True)
    
    def _generate_plots(self, trainer: L.Trainer, current_epoch: int, is_final: bool = False):
        """Generate training plots from CSV logs."""
        try:
            # Find the CSV log file
            csv_file = self._find_csv_log_file(trainer)
            
            if csv_file is None:
                logger.warning("No CSV log file found for plotting")
                return
            
            # Generate plots
            self._create_training_plots(csv_file, current_epoch, is_final)
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
    
    def _find_csv_log_file(self, trainer: L.Trainer) -> Path:
        """Find the current CSV log file."""
        if not hasattr(trainer.logger, 'log_dir'):
            return None
            
        log_dir = Path(trainer.logger.log_dir)
        metrics_file = log_dir / "metrics.csv"
        
        if metrics_file.exists():
            return metrics_file
        else:
            return None
    
    def _create_training_plots(self, csv_file: Path, current_epoch: int, is_final: bool = False):
        """Create training curves from CSV data."""
        try:
            # Read CSV data
            df = pd.read_csv(csv_file)
            
            if len(df) == 0:
                logger.warning("CSV file is empty, skipping plot generation")
                return
            
            logger.info(f"Generating plots from {len(df)} data points (epoch {current_epoch})")
            
            # Set up plotting style
            plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            epoch_suffix = f"_final" if is_final else f"_epoch_{current_epoch:03d}"
            fig.suptitle(f'Training Progress - Epoch {current_epoch}' + 
                        (" (Final)" if is_final else ""), fontsize=16)
            
            # Plot 1: Loss curves
            ax = axes[0, 0]
            loss_cols = [col for col in df.columns if 'loss' in col.lower()]
            for col in loss_cols:
                if not df[col].isna().all():
                    valid_data = df[df[col].notna()]
                    ax.plot(valid_data['epoch'], valid_data[col], 
                           label=col.replace('_', ' ').replace('/', ' ').title(), linewidth=2)
            
            ax.set_title('Loss Curves')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            if loss_cols:
                ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Performance metrics
            ax = axes[0, 1]
            perf_cols = [col for col in df.columns if any(term in col.lower() 
                         for term in ['acc', 'f1', 'auroc', 'ap', 'balanced'])]
            for col in perf_cols:
                if not df[col].isna().all():
                    valid_data = df[df[col].notna()]
                    ax.plot(valid_data['epoch'], valid_data[col], 
                           label=col.replace('_', ' ').replace('/', ' ').title(), linewidth=2)
            
            ax.set_title('Performance Metrics')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)  # Most metrics are 0-1
            if perf_cols:
                ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Learning rate
            ax = axes[1, 0]
            lr_cols = [col for col in df.columns if 'lr' in col.lower() or 'learning' in col.lower()]
            for col in lr_cols:
                if not df[col].isna().all():
                    valid_data = df[df[col].notna()]
                    ax.plot(valid_data['epoch'], valid_data[col], 
                           label='Learning Rate', linewidth=2)
            
            ax.set_title('Learning Rate')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            if lr_cols:
                ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Training vs Validation comparison
            ax = axes[1, 1]
            # Find paired train/val metrics
            train_cols = [col for col in df.columns if col.startswith('Train/')]
            val_cols = [col for col in df.columns if col.startswith('Val/')]
            
            # Plot a few key metrics
            key_metrics = ['loss', 'AP', 'F1']
            colors = ['red', 'blue', 'green']
            
            for i, metric in enumerate(key_metrics):
                train_col = f'Train/{metric}'
                val_col = f'Val/{metric}'
                
                if train_col in train_cols and not df[train_col].isna().all():
                    valid_data = df[df[train_col].notna()]
                    ax.plot(valid_data['epoch'], valid_data[train_col], 
                           color=colors[i], linestyle='-', label=f'Train {metric}', linewidth=2)
                
                if val_col in val_cols and not df[val_col].isna().all():
                    valid_data = df[df[val_col].notna()]
                    ax.plot(valid_data['epoch'], valid_data[val_col], 
                           color=colors[i], linestyle='--', label=f'Val {metric}', linewidth=2)
            
            ax.set_title('Train vs Validation')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plots
            if self.plot_format in ['png', 'both']:
                plot_path = self.save_dir / f'training_curves{epoch_suffix}.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved: {plot_path}")
            
            if self.plot_format in ['pdf', 'both']:
                plot_path = self.save_dir / f'training_curves{epoch_suffix}.pdf'
                plt.savefig(plot_path, bbox_inches='tight')
                logger.info(f"PDF saved: {plot_path}")
            
            if self.show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
            plt.close()  # Make sure to close the figure even on error
