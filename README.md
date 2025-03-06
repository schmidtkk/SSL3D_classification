# Reproducing the fine-tuning

## MRNet 
### Primus-M
Fine-tuning:

`python main.py env=cluster model=primus data=mrnet data.module.batch_size=8 trainer.accumulate_grad_batches=48 data.module.data_root_dir=<path/to/data> model.chpt_path=<path/to/checkpoint>`

Training from scratch:

`python main.py env=cluster model=primus data=mrnet data.module.batch_size=8 trainer.accumulate_grad_batches=48 data.module.data_root_dir=<path/to/data> model.pretrained=False`

### ResEnc-L
Fine-tuning:

`python main.py env=cluster model=resenc data=mrnet data.module.batch_size=16 trainer.accumulate_grad_batches=12 data.module.data_root_dir=<path/to/data> model.chpt_path=<path/to/checkpoint>`

Training from scratch:

`python main.py env=cluster model=resenc data=mrnet data.module.batch_size=16 trainer.accumulate_grad_batches=12  data.module.data_root_dir=<path/to/data> model.pretrained=False`

## RSNA Spine
### Primus-M
Fine-tuning:

`python main.py env=cluster model=primus data=rsna_spine data.module.batch_size=2 trainer.accumulate_grad_batches=192 data.module.data_root_dir=<path/to/data> model.chpt_path=<path/to/checkpoint>`

Training from scratch:

`python main.py env=cluster model=primus data=rsna_spine data.module.batch_size=2 trainer.accumulate_grad_batches=192 data.module.data_root_dir=<path/to/data> model.pretrained=False`

### ResEnc-L
Fine-tuning:

`python main.py env=cluster model=resenc data=rsna_spine data.module.batch_size=4 trainer.accumulate_grad_batches=48 data.module.data_root_dir=<path/to/data> model.chpt_path=<path/to/checkpoint>`

Training from scratch:

`python main.py env=cluster model=resenc data=rsna_spine data.module.batch_size=4 trainer.accumulate_grad_batches=48 data.module.data_root_dir=<path/to/data> model.pretrained=False`

## ABIDE
### Primus-M
Fine-tuning:

`python main.py env=cluster model=primus data=abide data.module.batch_size=2 trainer.accumulate_grad_batches=48 data.module.data_root_dir=<path/to/data> model.chpt_path=<path/to/checkpoint>`

Training from scratch:

`python main.py env=cluster model=primus data=abide data.module.batch_size=2 trainer.accumulate_grad_batches=48 data.module.data_root_dir=<path/to/data> model.pretrained=False`

### ResEnc-L
Fine-tuning:

`python main.py env=cluster model=resenc data=abide data.module.batch_size=4 trainer.accumulate_grad_batches=96 data.module.data_root_dir=<path/to/data> model.chpt_path=<path/to/checkpoint>`

Training from scratch:

`python main.py env=cluster model=resenc data=abide data.module.batch_size=4 trainer.accumulate_grad_batches=96 data.module.data_root_dir=<path/to/data> model.pretrained=False`

# Installation
## Requirements
Install the requirements in a [virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) by:

```shell
pip install -r requirements.txt
```

You might need to adapt the cuda versions for torch and torchvision. 
Find a torch installation guide for your system [here](https://pytorch.org/get-started/locally/). 


