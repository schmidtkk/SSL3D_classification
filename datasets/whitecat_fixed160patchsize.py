import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .base_datamodule import BaseDataModule
from .blosc2io import Blosc2IO


class whitecat_fixed160patchsize_Data(Dataset):
    def __init__(self, root, split, fold, transform=None):
        super().__init__()
        """
        whitecat_median_shape Dataset
        """
        self.img_dir = Path(root) / "WhiteCat_fixed160patchsize"
        label_file = Path(root) / "WhiteCat_fixed160patchsize/labels.json"
        split_file = Path(root) / "WhiteCat_fixed160patchsize/splits.json"

        with open(split_file) as f:
            self.img_files = json.load(f)[fold]["train" if split == "train" else "val"]

        with open(label_file) as f:
            labels = json.load(f)
        self.labels = [labels[i] for i in self.img_files]

        self.transform = transform

    def __getitem__(self, idx):

        img1, _ = Blosc2IO.load(self.img_dir / (self.img_files[idx] + "_01.b2nd"), mode="r")
        img2, _ = Blosc2IO.load(self.img_dir / (self.img_files[idx] + "_02.b2nd"), mode="r")
        img =  np.stack([img1[0], img2[0]], axis=0)

        if self.transform:
            img = self.transform(**{"image": torch.from_numpy(img[...])})["image"]
        else:
            img = torch.from_numpy(img[...])

        return img, self.labels[idx]

    def __len__(self):
        return len(self.img_files)


class whitecat_fixed160patchsize_DataModule(BaseDataModule):
    def __init__(self, **params):
        super(whitecat_fixed160patchsize_DataModule, self).__init__(**params)

    def setup(self, stage: str):

        self.train_dataset = whitecat_fixed160patchsize_Data(
            self.data_path,
            split="train",
            transform=self.train_transforms,
            fold=self.fold,
        )
        self.val_dataset = whitecat_fixed160patchsize_Data(
            self.data_path,
            split="val",
            transform=self.test_transforms,
            fold=self.fold,
        )