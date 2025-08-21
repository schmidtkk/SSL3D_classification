# SSL3D Classification Framework - CSV Logging & Auto-Plotting Guide

## Overview

This framework now uses **CSV logging by default** instead of wandb, with **automatic plot generation** during training. This provides a simple, local logging solution that doesn't require external services.

## Key Features

- 🔄 **CSV Logging**: All metrics saved to local CSV files
- 📊 **Auto-Plotting**: Training curves generated every N epochs (default: 10)
- 🎯 **No External Dependencies**: Pure local logging, no wandb/cloud services needed
- 📈 **Real-time Monitoring**: View training progress without stopping training

## Quick Start

### Basic Training Command

```bash
cd /data/weidong/workspace/SSL3D_classification

# Train with default CSV logging and auto-plotting
python main.py env=local model=primus data=fomo_task1 trainer.devices=1
```

### Custom Training Examples

```bash
# Train for 100 epochs with plots every 5 epochs
python main.py env=local model=primus data=fomo_task1 trainer.max_epochs=100 trainer.callbacks.auto_plot.plot_every_n_epochs=5

# Disable auto-plotting (CSV logging still active)
python main.py env=local model=primus data=fomo_task1 ~trainer.callbacks.auto_plot

# Change plot format to PDF
python main.py env=local model=primus data=fomo_task1 trainer.callbacks.auto_plot.plot_format=pdf

# Save plots to custom directory
python main.py env=local model=primus data=fomo_task1 trainer.callbacks.auto_plot.save_dir=./my_plots
```

## File Structure

After training, your experiment directory will contain:

```
experiments/
└── debug/classification/fomo_task1/
    ├── fomo_task1/              # CSV logger directory
    │   └── version_0/
    │       ├── metrics.csv      # Raw training metrics
    │       ├── hparams.yaml     # Hyperparameters
    │       └── events.out.tfevents.* # TensorBoard events (if enabled)
    └── plots/                   # Auto-generated plots
        ├── training_curves_epoch_010.png
        ├── training_curves_epoch_020.png
        └── training_curves_final.png
```

## Configuration Options

### CSV Logger Settings

The CSV logger is now the default logger. You can customize it:

```yaml
trainer:
  logger:
    _target_: lightning.pytorch.loggers.CSVLogger
    save_dir: ${exp_dir}/${teamname}/classification/${data.module.name}
    name: ${data.module.name}
    version: null  # Auto-increment versions
```

### Auto-Plot Callback Settings

```yaml
trainer:
  callbacks:
    auto_plot:
      _target_: callbacks.auto_plot.AutoPlotCallback
      plot_every_n_epochs: 10        # Plot frequency (default: 10)
      save_dir: ./plots              # Where to save plots
      plot_format: png               # 'png', 'pdf', or 'both'
      show_plots: False              # Display plots (set True for interactive)
```

## Manual Plotting

You can also generate plots manually at any time:

```bash
# Plot from latest CSV logs
python plot_csv_logs.py --log_dir experiments/debug/classification/fomo_task1

# Plot from specific version
python plot_csv_logs.py --log_dir experiments/debug/classification/fomo_task1/fomo_task1/version_0

# Custom output directory
python plot_csv_logs.py --log_dir experiments/debug/classification/fomo_task1 --output_dir ./custom_plots
```

## Example Datasets

python main.py env=local model=primus data=fomo_task1 trainer.devices=1### FOMO Multi-Task Dataset (Enhanced XYZ Format)

```bash
# Task 1: Infarct Detection (4 channels: ADC, DWI, FLAIR, SWI)
python main.py env=local model=primus data=fomo_task1 trainer.devices=1

# Task 2: Meningioma Segmentation (3 channels: DWI, FLAIR, SWI)
python main.py env=local model=primus data=fomo_task2_meningioma trainer.devices=1

# Task 3: Brain Age Regression (2 channels: T1w, T2w)
python main.py env=local model=primus data=fomo_task3_brainage trainer.devices=1

# Full training (300 epochs) with plots every 20 epochs
python main.py env=local model=primus data=fomo_task1 \
    trainer.devices=1 \
    trainer.callbacks.auto_plot.plot_every_n_epochs=20

# Training with custom batch size and plotting
python main.py env=local model=primus data=fomo_task1 \
    data.module.batch_size=4 \
    trainer.callbacks.auto_plot.plot_every_n_epochs=5 \
    trainer.callbacks.auto_plot.plot_format=both
```

### ABIDE Dataset

```bash
# Train ABIDE with CSV logging
python main.py env=local model=primus data=abide trainer.devices=1

# ABIDE with frequent plotting
python main.py env=local model=primus data=abide \
    trainer.callbacks.auto_plot.plot_every_n_epochs=5
```

### Custom Datasets

To use CSV logging with your custom dataset:

1. **Create dataset config** (e.g., `cli_configs/data/my_dataset.yaml`):
```yaml
# @package _global_
data:
  module:
    _target_: datasets.my_dataset.MyDataModule
    name: my_dataset
    data_root_dir: ${data_dir}
    batch_size: 4
  num_classes: 2
  patch_size: [64, 64, 64]

model:
  task: 'Classification'
  input_channels: 1
  input_shape: ${data.patch_size}

trainer:
  max_epochs: 100
  # CSV logging is now default - no need to specify
  
metrics:
  - 'f1'
  - 'balanced_acc'
```

2. **Train with your dataset**:
```bash
python main.py env=local model=primus data=my_dataset trainer.devices=1
```

## Monitoring Training Progress

### Real-time Monitoring

1. **Watch CSV logs**:
```bash
# Monitor latest metrics
tail -f experiments/debug/classification/fomo_task1/fomo_task1/version_0/metrics.csv
```

2. **Check auto-generated plots**:
```bash
# View latest plots
ls -la experiments/debug/classification/fomo_task1/plots/
```

### Post-training Analysis

1. **Generate final summary**:
```bash
python plot_csv_logs.py --log_dir experiments/debug/classification/fomo_task1
# Creates: training_summary.csv with best metrics
```

2. **Compare multiple runs**:
```bash
# Plot different versions
python plot_csv_logs.py --log_dir experiments/debug/classification/fomo_task1/fomo_task1/version_0
python plot_csv_logs.py --log_dir experiments/debug/classification/fomo_task1/fomo_task1/version_1
```

## Advanced Usage

### Switching Back to Wandb

If you need wandb for a specific run:

```bash
python main.py env=local model=primus data=fomo_task1 \
    trainer.logger._target_=lightning.pytorch.loggers.WandbLogger \
    trainer.logger.project=my_project \
    trainer.logger.offline=False \
    ~trainer.callbacks.auto_plot  # Disable auto-plot for wandb
```

### Combining CSV and TensorBoard

```bash
# Use both CSV and TensorBoard
python main.py env=local model=primus data=fomo_task1 \
    ++trainer.logger_tb._target_=lightning.pytorch.loggers.TensorBoardLogger \
    ++trainer.logger_tb.save_dir='${exp_dir}/${teamname}/tensorboard' \
    ++trainer.logger_tb.name=${data.module.name}
```

### Custom Plot Styling

Modify `callbacks/auto_plot.py` to customize plot appearance:

```python
# In _create_training_plots method
plt.style.use('dark_background')  # Dark theme
plt.rcParams['font.size'] = 12    # Larger fonts
plt.rcParams['figure.dpi'] = 150  # Higher DPI
```

## Troubleshooting

### Common Issues

1. **Plots not generating**:
   - Check if matplotlib is installed: `pip install matplotlib pandas`
   - Verify CSV log file exists in logger directory
   - Check callback is enabled in configuration

2. **Permission errors**:
   - Ensure write permissions to plot directory
   - Check disk space availability

3. **Memory issues with large plots**:
   - Reduce `plot_every_n_epochs` value
   - Use 'png' instead of 'both' format
   - Close plots after viewing: `plt.close()`

### Debug Mode

Enable debug logging:

```bash
# Run with debug logging
PYTHONPATH=. python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
" main.py env=local model=primus data=fomo_task1
```

## Performance Tips

1. **Optimize plotting frequency**:
   - Use `plot_every_n_epochs=20` for long training (>200 epochs)
   - Use `plot_every_n_epochs=5` for short experiments

2. **Storage considerations**:
   - PNG files: ~100KB per plot
   - PDF files: ~200KB per plot
   - CSV logs: ~1KB per epoch

3. **Headless training**:
   - Set `show_plots: False` for cluster/remote training
   - Use `plot_format: png` for faster generation

## Migration from Wandb

If migrating existing configs from wandb to CSV logging:

1. **Remove wandb-specific parameters**:
```yaml
# OLD (wandb)
trainer:
  logger:
    _target_: lightning.pytorch.loggers.WandbLogger
    project: my_project
    offline: False
    group: experiment_1

# NEW (CSV) - these are now defaults
trainer:
  # CSV logger is default, auto-plot callback included
  max_epochs: 100
```

2. **Update plotting workflow**:
```bash
# OLD: Check wandb dashboard online
# NEW: Check local plots
ls experiments/debug/classification/my_dataset/plots/
```

## Support

For questions or issues:
- Check existing CSV logs: `experiments/debug/classification/[dataset_name]/`
- Review plot generation logs in terminal output
- Verify callback configuration in printed config before training starts

---

**Happy Training! 📈🎯**
