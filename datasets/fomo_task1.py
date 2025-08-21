"""
FOMO Multi-Task Dataset Classes - XYZ Format (NNSSL Compatible)
Enhanced with flexible modal                # Get subject list for this fold and split
        fold_data = cv_data[str(fold)]  # Use string key since splits.json uses string keys
        if split == 'train':
            self.subjects = fold_data['train']
        elif split in ['val', 'valid', 'validation']:
            self.subjects = fold_data['val']
        elif split == 'test':
            # If no test split in file, use validation for now
            self.subjects = fold_data.get('test', fold_data['val'])
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Get subject list for this fold and split
        fold_data = cv_data[str(fold)]  # Use string key since splits.json uses string keys
        if split == 'train':
            self.subjects = fold_data['train']
        elif split in ['val', 'valid', 'validation']:
            self.subjects = fold_data['val']
        elif split == 'test':
            # If no test split in file, use validation for now
            self.subjects = fold_data.get('test', fold_data['val'])
        else:
            raise ValueError(f"Unknown split: {split}")t for all FOMO tasks
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import blosc2
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from datasets.base_datamodule import BaseDataModule

logger = logging.getLogger(__name__)

class FomoDataXYZ(Dataset):
    """
    Enhanced FOMO Dataset for XYZ format with flexible modality support
    
    Supports all FOMO tasks:
    - Task 1: Infarct Detection (ADC, DWI, FLAIR, SWI)
    - Task 2: Meningioma Segmentation (DWI, FLAIR, SWI)
    - Task 3: Brain Age Regression (T1w, T2w)
    
    Data format: (X, Y, Z) = (256, 256, 32)
    Spacing: (0.45, 0.45, 5.0) mm in XYZ order
    """
    
    def __init__(
        self,
        data_root_dir: str,
        split: str = 'train',
        fold: int = 0,
        transforms: Optional[callable] = None,
        modalities: Optional[List[str]] = None,
        task: str = 'task1'
    ):
        """
        Initialize FOMO dataset with flexible modality support
        
        Args:
            data_root_dir: Path to preprocessed data directory
            split: 'train', 'val', or 'test'
            fold: Cross-validation fold (0-based)
            transforms: Optional transforms to apply
            modalities: List of modalities to load. If None, uses task defaults
            task: Task identifier for default modality selection
        """
        super().__init__()
        
        self.data_root_dir = Path(data_root_dir)
        self.split = split
        self.fold = fold
        self.transforms = transforms
        self.task = task
        
        # Task-specific modality defaults
        self.task_modalities = {
            'task1': ['adc', 'dwi_b1000', 'flair', 'swi'],      # Infarct Detection
            'task2': ['dwi_b1000', 'flair', 'swi'],             # Meningioma (no ADC) 
            'task3': ['t1w', 't2w'],                            # Brain Age Regression
            'infarct': ['adc', 'dwi_b1000', 'flair', 'swi'],    # Alias for task1
            'meningioma': ['dwi_b1000', 'flair', 'swi'],        # Alias for task2
            'brainage': ['t1w', 't2w']                          # Alias for task3
        }
        
        # Set modalities (user override or task default)
        if modalities is not None:
            self.modalities = modalities
        elif task in self.task_modalities:
            self.modalities = self.task_modalities[task]
        else:
            # Fallback to task1 default
            logger.warning(f"Unknown task '{task}', using task1 modalities")
            self.modalities = self.task_modalities['task1']
            
        logger.info(f"Using modalities for {task}: {self.modalities}")
        
        # Validate modality support
        supported_modalities = ['adc', 'dwi_b1000', 'flair', 'swi', 't1w', 't2w', 't2star']
        unsupported = set(self.modalities) - set(supported_modalities)
        if unsupported:
            logger.warning(f"Unsupported modalities detected: {unsupported}")
            
        self.num_channels = len(self.modalities)
        
        # Load cross-validation splits
        cv_file = self.data_root_dir / "splits.json"
        with open(cv_file, 'r') as f:
            cv_data = json.load(f)
        
        # Get subject list for this fold and split
        fold_data = cv_data[str(fold)]  # Use string key since splits.json uses string keys
        if split == 'train':
            self.subjects = fold_data['train']
        elif split in ['val', 'valid', 'validation']:
            self.subjects = fold_data['val']
        elif split == 'test':
            self.subjects = fold_data.get('test', fold_data['val'])  # Use val if no test
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Load labels
        labels_file = self.data_root_dir / "labels.json"
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)
        
        # Analyze dataset composition
        self._analyze_dataset()
        
        logger.info(f"FomoDataXYZ: {split} split, fold {fold}, {len(self.subjects)} samples, {self.num_channels} channels")
        
    def _get_modality_suffix(self, modality: str) -> str:
        """Get the suffix for a modality used in preprocessing"""
        suffix_map = {
            'adc': '0000',
            'dwi_b1000': '0001', 
            'flair': '0002',
            'swi': '0003'
        }
        return suffix_map.get(modality, '0000')
        
    def _analyze_dataset(self):
        """Analyze dataset composition and missing data"""
        total_files = 0
        missing_files = 0
        modality_counts = {mod: 0 for mod in self.modalities}
        
        for subject in self.subjects:
            for modality in self.modalities:
                file_path = self.data_root_dir / f"{subject}_{self._get_modality_suffix(modality)}.b2nd"
                if file_path.exists():
                    # Check if it's a real file or zero-padded (missing)
                    try:
                        data = blosc2.open(str(file_path))[:]
                        if np.any(data != 0):  # Not all zeros
                            modality_counts[modality] += 1
                        else:
                            missing_files += 1
                    except:
                        missing_files += 1
                else:
                    missing_files += 1
                total_files += 1
        
        # Print statistics
        for modality, count in modality_counts.items():
            availability = count / len(self.subjects) * 100
            logger.info(f"  {modality}: {count}/{len(self.subjects)} available ({availability:.1f}%)")
        
        if missing_files > 0:
            logger.warning(f"Warning: {missing_files} missing modality files detected")
    
    def __len__(self) -> int:
        return len(self.subjects)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample
        
        Returns:
            data: Tensor of shape (C, X, Y, Z) where C = len(self.modalities)
            label: Integer label
        """
        subject = self.subjects[idx]
        
        # Load each modality
        modality_data = []
        
        for modality in self.modalities:
            file_path = self.data_root_dir / f"{subject}_{self._get_modality_suffix(modality)}.b2nd"
            
            try:
                if file_path.exists():
                    data = blosc2.open(str(file_path))[:]  # Shape: (Z, Y, X) from preprocessing
                    
                    # Handle batch dimension from preprocessing
                    if data.shape == (1, 32, 256, 256):  # (Batch, Z, Y, X) from preprocessing
                        data = data[0]  # Remove batch dimension: (32, 256, 256)
                    
                    # Convert from ZYX format (32, 256, 256) to XYZ format (256, 256, 32)
                    if data.shape == (32, 256, 256):  # Expected ZYX format from preprocessing
                        # Transpose from (Z, Y, X) to (X, Y, Z)
                        data = data.transpose(2, 1, 0)  # (32, 256, 256) -> (256, 256, 32)
                    else:
                        logger.warning(f"Unexpected shape for {subject}/{modality}: {data.shape}, expected (32, 256, 256)")
                        # Create zero data in XYZ format if shape is wrong
                        data = np.zeros((256, 256, 32), dtype=np.float32)
                    
                    # Check if this is actually missing data (all zeros)
                    if not np.any(data):
                        logger.debug(f"Missing modality {modality} for {subject}, using zeros")
                    
                else:
                    logger.debug(f"Missing modality {modality} for {subject}, using zeros")
                    data = np.zeros((256, 256, 32), dtype=np.float32)  # XYZ format
                    
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
                data = np.zeros((256, 256, 32), dtype=np.float32)  # XYZ format
            
            modality_data.append(data)
        
        # Stack modalities along channel dimension: (C, X, Y, Z)
        combined_data = np.stack(modality_data, axis=0)
        
        # Convert to tensor and ensure correct format
        data_tensor = torch.from_numpy(combined_data).float()
        
        # Verify final shape
        expected_final_shape = (self.num_channels, 256, 256, 32)  # (C, X, Y, Z) 
        if data_tensor.shape != expected_final_shape:
            logger.error(f"Final tensor shape {data_tensor.shape} != expected {expected_final_shape}")
            
        # Apply transforms if provided
        if self.transforms:
            # batchgenerators transforms expect numpy arrays
            data_numpy = data_tensor.numpy() if isinstance(data_tensor, torch.Tensor) else data_tensor
            data_dict = {'data': data_numpy[None, ...]}  # Add batch dimension: (1, C, X, Y, Z)
            transformed_dict = self.transforms(**data_dict)
            data_tensor = torch.from_numpy(transformed_dict['data'][0]).float()  # Back to tensor: (C, X, Y, Z)
        
        # Get label
        label = self.labels[subject]
        
        return data_tensor, label
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution for this split"""
        class_counts = {}
        for subject in self.subjects:
            label = self.labels[subject]
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts
    
    def get_modality_info(self) -> Dict[str, any]:
        """Get information about modalities used"""
        return {
            'modalities': self.modalities,
            'num_channels': self.num_channels,
            'task': self.task,
            'supported_modalities': ['adc', 'dwi_b1000', 'flair', 'swi', 't1w', 't2w', 't2star']
        }

class FomoDataModuleXYZ(BaseDataModule):
    """
    Enhanced Lightning DataModule for FOMO tasks with flexible modality support
    """
    
    def __init__(
        self,
        data_root_dir: str,
        name: str = "fomo_task1",
        batch_size: int = 2,
        train_transforms=None,
        test_transforms=None,
        random_batches: bool = False,
        num_workers: int = 0,
        prepare_data_per_node: bool = False,
        fold: int = 0,
        modalities: Optional[List[str]] = None,
        task: str = 'task1',
        **kwargs
    ):
        super().__init__(
            data_root_dir=data_root_dir,
            name=name,
            batch_size=batch_size,
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            random_batches=random_batches,
            num_workers=num_workers,
            prepare_data_per_node=prepare_data_per_node,
            fold=fold,
            **kwargs
        )
        
        self.modalities = modalities
        self.task = task
        
        # Determine number of input channels
        task_modalities = {
            'task1': ['adc', 'dwi_b1000', 'flair', 'swi'],      # 4 channels
            'task2': ['dwi_b1000', 'flair', 'swi'],             # 3 channels 
            'task3': ['t1w', 't2w'],                            # 2 channels
            'infarct': ['adc', 'dwi_b1000', 'flair', 'swi'],    # 4 channels
            'meningioma': ['dwi_b1000', 'flair', 'swi'],        # 3 channels
            'brainage': ['t1w', 't2w']                          # 2 channels
        }
        
        effective_modalities = modalities or task_modalities.get(task, task_modalities['task1'])
        self.num_channels = len(effective_modalities)
        
        logger.info(f"FomoDataModuleXYZ: Task {task}, {self.num_channels} channels, modalities: {effective_modalities}")
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets"""
        if stage == 'fit' or stage is None:
            self.train_dataset = FomoDataXYZ(
                data_root_dir=self.data_path,  # Use parent class attribute
                split='train',
                fold=self.fold,
                transforms=self.train_transforms,
                modalities=self.modalities,
                task=self.task
            )
            
            self.val_dataset = FomoDataXYZ(
                data_root_dir=self.data_path,  # Use parent class attribute
                split='val',
                fold=self.fold,
                transforms=self.test_transforms,
                modalities=self.modalities,
                task=self.task
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = FomoDataXYZ(
                data_root_dir=self.data_path,  # Use parent class attribute
                split='test',
                fold=self.fold,
                transforms=self.test_transforms,
                modalities=self.modalities,
                task=self.task
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_input_channels(self) -> int:
        """Get number of input channels for model initialization"""
        return self.num_channels
    
    def get_task_info(self) -> Dict[str, any]:
        """Get task information"""
        return {
            'task': self.task,
            'num_channels': self.num_channels,
            'modalities': self.modalities,
            'data_format': 'XYZ (NNSSL compatible)',
            'input_shape': (self.num_channels, 256, 256, 32)
        }

# Aliases for backward compatibility and task-specific convenience
FomoTask1DataXYZ = FomoDataXYZ  # Backward compatibility
FomoTask1DataModuleXYZ = FomoDataModuleXYZ  # Backward compatibility

class FomoInfarctDataXYZ(FomoDataXYZ):
    """Task 1: Infarct Detection (ADC, DWI, FLAIR, SWI)"""
    def __init__(self, **kwargs):
        kwargs['task'] = 'infarct'
        super().__init__(**kwargs)

class FomoMeningiomaDataXYZ(FomoDataXYZ):
    """Task 2: Meningioma Segmentation (DWI, FLAIR, SWI)"""
    def __init__(self, **kwargs):
        kwargs['task'] = 'meningioma'
        super().__init__(**kwargs)

class FomoBrainAgeDataXYZ(FomoDataXYZ):
    """Task 3: Brain Age Regression (T1w, T2w)"""
    def __init__(self, **kwargs):
        kwargs['task'] = 'brainage'
        super().__init__(**kwargs)

class FomoInfarctDataModuleXYZ(FomoDataModuleXYZ):
    """Task 1: Infarct Detection DataModule"""
    def __init__(self, **kwargs):
        kwargs['task'] = 'infarct'
        super().__init__(**kwargs)

class FomoMeningiomaDataModuleXYZ(FomoDataModuleXYZ):
    """Task 2: Meningioma Segmentation DataModule"""
    def __init__(self, **kwargs):
        kwargs['task'] = 'meningioma'
        super().__init__(**kwargs)

class FomoBrainAgeDataModuleXYZ(FomoDataModuleXYZ):
    """Task 3: Brain Age Regression DataModule"""
    def __init__(self, **kwargs):
        kwargs['task'] = 'brainage'
        super().__init__(**kwargs)
