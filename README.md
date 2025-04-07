# 3D medical image classification repository

Welcome to this 3D medical image classification repository. The repository builds up on the [IMAGE CLASSIFICATION FRAMEWORK BY HELMHOLTZ IMAGING](https://github.com/MIC-DKFZ/image_classification)
This repository was extended to allow fine-tuning checkpoints from this repository: [nnssl](https://github.com/MIC-DKFZ/nnssl). 
# Installation
## Requirements
Install the requirements in a [virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) by:

```shell
pip install -r requirements.txt
```

You might need to adapt the cuda versions for torch and torchvision.
Find a torch installation guide for your system [here](https://pytorch.org/get-started/locally/).

Please also install the  [nnssl](https://github.com/MIC-DKFZ/nnssl) repository in the same environment!

# Dataset preprocessing
Currently, preprocessing is highly dataset- and user-dependent. However, in this file, we provide an example of how it can be done.
However in `./datasets/preocess3D_data/datasets/` you can find examples of how a dataset can be preprocessed. 

# Including other datasets

For including your own dataset follow these steps:
1. In the ```dataset``` directory create a new file that implements the [torch dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files) class for your data.
2. Additionally, create the [DataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) for your dataset by writing a class that inherits from `BaseDataModule`. Write the `init` and `setup` functions for your dataset. The dataloaders are already defined by the `BaseDataModule`. An example could look like this:
    ```python
    from .base_datamodule import BaseDataModule

    class CustomDataModule(BaseDataModule):
      def __init__(self, **params):
          super(CustomDataModule, self).__init__(**params)

      def setup(self, stage: str):
          self.train_dataset = YourCustomPytorchDataset(
              data_path=self.data_path,
              split="train",
              transform=self.train_transforms,
          )
          self.val_dataset = YourCustomPytorchDataset(
              data_path=self.data_path,
              split="val",
              transform=self.test_transforms,
          )
    ```
   Note that the `__init__` function takes `**params` and passes them to the super init. By doing so the attributes `self.data_path`, `self.train_transforms` and `self.test_transforms` are already set automatically and can be used in the `setup` function. The `self.data_path` is a joined path consisting of the configs `data.module.data_root_dir` and `data.module.name`.
   Custom transforms can be added in `./augmentation/policies/<your-data>.py`. They need to inherit from the `BaseTransform` class. See the existing transforms for examples! 
3. Add a `<your-data>.yaml` file to the data config group, defining some data-specific variables. For CIFAR-10 it looks like this:
    ```yaml
    # @package _global_
    data:
      module:
        _target_: datasets.abide.AbideDataModule
        name: ABIDE
        data_root_dir: ???
        batch_size: 4
        train_transforms:
        _target_: augmentation.policies.batchgenerators.get_training_transforms
        patch_size: ${data.patch_size}
        rotation_for_DA: 0.523599
        mirror_axes: [0,1,2]
        do_dummy_2d_data_aug: False
        test_transforms: null
      cv:
        k:5

      num_classes: 2
      patch_size: [160, 192, 224]

    model:
      task: 'Classification'
      cifar_size: False
      input_channels: 1
      input_dim: 3
      input_shape: ${data.patch_size}
      optimizer: AdamW
      lr: 0.0001
      warmstart: 20
      weight_decay: 1e-2
   
   trainer:
    logger:
      project: ABIDE
    accumulate_grad_batches: 96
    max_epochs: 200
   
   metrics:
    - 'f1'
    - 'balanced_acc'
    - 'ap'
    - 'auroc'
    ```
   The `data.module._target_` defines the path to your `DataModule`. Note that the first line of the file needs to be `# @package _global_` in order for Hydra to read the config properly.


# Training 
### Primus-M
Fine-tuning:

`python main.py env=cluster model=primus data=Datasetname data.module.batch_size=8 trainer.accumulate_grad_batches=48 data.module.data_root_dir=<path/to/data> model.chpt_path=<path/to/checkpoint>`

Training from scratch:

`python main.py env=cluster model=primus data=Datasetname data.module.batch_size=8 trainer.accumulate_grad_batches=48 data.module.data_root_dir=<path/to/data> model.pretrained=False`

### ResEnc-L
Fine-tuning:

`python main.py env=cluster model=resenc data=Datasetname data.module.batch_size=16 trainer.accumulate_grad_batches=12 data.module.data_root_dir=<path/to/data> model.chpt_path=<path/to/checkpoint>`

Training from scratch:

`python main.py env=cluster model=resenc data=Datasetname data.module.batch_size=16 trainer.accumulate_grad_batches=12  data.module.data_root_dir=<path/to/data> model.pretrained=False`






