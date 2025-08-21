"""
Preprocessing pipeline for FOMO Task1 dataset
Handles anisotropic data with target spacing (0.45, 0.45, 5.0) and patch size 256x256x32
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from datasets.preprocess_3D_data.crop_to_mask import load_image_np, crop_center_with_padding_np, get_mask_center
from datasets.preprocess_3D_data.default_resampling import resample_data_or_seg_to_spacing, resample_data_or_seg_to_shape
import numpy as np
from datasets.preprocess_3D_data.cross_validation import generate_crossval_split
from multiprocessing import Pool
import SimpleITK as sitk
from datasets.preprocess_3D_data.blosc_helper import save_case, comp_blosc2_params
from batchgenerators.utilities.file_and_folder_operations import *
from datasets.preprocess_3D_data.normalization import ZScoreNormalization
from typing import Tuple, List, Dict
from pathlib import Path
import json
import glob


def organize_raw_data(
    raw_data_path: str,
    organized_path: str
) -> Dict[str, int]:
    """
    Organize FOMO Task1 data into nnU-Net format and extract labels
    
    Args:
        raw_data_path: Path to /data/weidong/fomo_finetune/fomo-task1
        organized_path: Path where organized data should be stored
    
    Returns:
        Dictionary mapping subject IDs to labels
    """
    raw_path = Path(raw_data_path)
    org_path = Path(organized_path)
    
    # Create organized directory structure
    images_dir = org_path / "imagesTr"
    maybe_mkdir_p(images_dir)
    
    label_dict = {}
    modality_mapping = {
        'adc': '0000',
        'dwi_b1000': '0001', 
        'flair': '0002',
        'swi': '0003'
    }
    
    print("Organizing raw data into nnU-Net format...")
    
    # Process each subject
    for subject_dir in sorted(raw_path.glob("preprocessed/sub_*")):
        subject_id = subject_dir.name
        session_dir = subject_dir / "ses_1"
        
        if not session_dir.exists():
            print(f"Warning: No ses_1 directory found for {subject_id}")
            continue
        
        # Copy each modality
        modalities_found = []
        for modality, suffix in modality_mapping.items():
            src_file = session_dir / f"{modality}.nii.gz"
            dst_file = images_dir / f"{subject_id}_{suffix}.nii.gz"
            
            if src_file.exists():
                # Read and write to ensure consistency
                img = sitk.ReadImage(str(src_file))
                sitk.WriteImage(img, str(dst_file))
                modalities_found.append(modality)
                print(f"Organized {subject_id} - {modality}")
            else:
                print(f"Warning: Missing {modality} for {subject_id}")
        
        # Extract label
        label_file = raw_path / "labels" / subject_id / "ses_1" / "label.txt"
        if label_file.exists():
            with open(label_file, 'r') as f:
                label = int(f.read().strip())
                label_dict[subject_id] = label
        else:
            print(f"Warning: No label found for {subject_id}")
    
    print(f"Organized {len(label_dict)} subjects with labels")
    print(f"Class distribution: {np.bincount(list(label_dict.values()))}")
    
    return label_dict


def run_case_anisotropic(
    base_path: str,
    img_id: str,
    target_spacing: Tuple[float, float, float],
    crop_size: Tuple[int, int, int],
    output_dir: str,
    modality_suffixes: List[str] = ['0000', '0001', '0002', '0003']
) -> None:
    """
    Preprocess a single case with anisotropic resampling
    
    Args:
        base_path: Base directory containing 'imagesTr'
        img_id: Subject identifier (e.g., 'sub_1')
        target_spacing: Target spacing (5.0, 0.45, 0.45) - in (z, y, x) order
        crop_size: Target patch size (32, 256, 256) - in (z, y, x) order
        output_dir: Output directory for preprocessed data
        modality_suffixes: List of modality suffixes ['0000', '0001', '0002', '0003']
    """
    print(f"Processing {img_id}...")
    
    # Load first modality to get reference spacing and determine crop center
    base_img_path = join(base_path, 'imagesTr', f'{img_id}_0000.nii.gz')
    if not os.path.exists(base_img_path):
        print(f"Error: Base image not found: {base_img_path}")
        return
    
    # Load reference image to get spacing and determine cropping center
    ref_img = sitk.ReadImage(base_img_path)
    ref_data = sitk.GetArrayFromImage(ref_img)[np.newaxis, ...]
    # SimpleITK spacing is (x, y, z), but numpy array is (z, y, x)
    # So we need to reverse the spacing to match the array order
    original_spacing = ref_img.GetSpacing()[::-1]  # Convert from (x, y, z) to (z, y, x)
    
    print(f"  Original spacing (z,y,x): {original_spacing}")
    print(f"  Target spacing (z,y,x): {target_spacing}")
    print(f"  Original shape (batch,z,y,x): {ref_data.shape}")
    
    # Resample reference to target spacing to determine center
    ref_resampled = resample_data_or_seg_to_spacing(ref_data, original_spacing, target_spacing)
    print(f"  Resampled shape: {ref_resampled.shape}")
    
    # Use center of resampled image for cropping
    center = np.array(ref_resampled.shape[1:]) // 2
    print(f"  Crop center: {center}")
    
    # Process each modality
    processed_data = []
    for i, suffix in enumerate(modality_suffixes):
        img_path = join(base_path, 'imagesTr', f'{img_id}_{suffix}.nii.gz')
        
        if not os.path.exists(img_path):
            print(f"  Warning: Missing modality {suffix} for {img_id}")
            continue
        
        # Load image
        img = sitk.ReadImage(img_path)
        data = sitk.GetArrayFromImage(img)[np.newaxis, ...]
        
        # Resample to target spacing (anisotropic)
        data_resampled = resample_data_or_seg_to_spacing(data, original_spacing, target_spacing)
        
        # Crop to target size around center
        cropped_data = crop_center_with_padding_np(data_resampled[0], center, crop_size)
        cropped_data = cropped_data[np.newaxis, ...]
        
        print(f"  Modality {suffix}: {data.shape} -> {data_resampled.shape} -> {cropped_data.shape}")
        
        # Normalize (z-score normalization within brain region)
        normalizer = ZScoreNormalization()
        # For brain data, we might want to use a simple brain mask
        # For now, use the data itself to compute normalization stats
        mask = cropped_data > np.percentile(cropped_data, 1)  # Simple intensity-based mask
        normalizer.use_mask_for_norm = True
        normalized_data = normalizer.run(cropped_data, mask.astype(np.uint8))
        
        # Save individual modality
        block_size_data, chunk_size_data = comp_blosc2_params(normalized_data.shape, crop_size, normalized_data.itemsize)
        out_path_truncated = join(output_dir, f'{img_id}_{suffix}')
        save_case(normalized_data, out_path_truncated, chunks=chunk_size_data, blocks=block_size_data)
        
        processed_data.append(normalized_data)
    
    print(f"  Completed {img_id} - processed {len(processed_data)} modalities")


def preprocess_fomo_task1_dataset(
    raw_data_dir: str,
    output_dir: str,
    target_spacing: Tuple[float, float, float] = (5.0, 0.45, 0.45),
    patch_size: Tuple[int, int, int] = (32, 256, 256),
    num_workers: int = 1
) -> None:
    """
    Complete preprocessing pipeline for FOMO Task1 dataset
    
    Args:
        raw_data_dir: Path to raw FOMO Task1 data
        output_dir: Output directory for preprocessed data
        target_spacing: Target anisotropic spacing (z, y, x)
        patch_size: Target anisotropic patch size (z, y, x)
        num_workers: Number of parallel workers
    """
    print("Starting FOMO Task1 preprocessing pipeline...")
    print(f"Raw data: {raw_data_dir}")
    print(f"Output: {output_dir}")
    print(f"Target spacing (z,y,x): {target_spacing}")
    print(f"Patch size (z,y,x): {patch_size}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Organize raw data
    temp_organized_dir = join(output_dir, "temp_organized")
    label_dict = organize_raw_data(raw_data_dir, temp_organized_dir)
    
    if not label_dict:
        raise RuntimeError("No subjects found with valid labels")
    
    # Step 2: Get list of subjects to process
    subjects = list(label_dict.keys())
    print(f"Found {len(subjects)} subjects to process")
    
    # Step 3: Process each subject
    modality_suffixes = ['0000', '0001', '0002', '0003']  # adc, dwi_b1000, flair, swi
    
    if num_workers == 1:
        # Sequential processing for debugging
        for subject_id in subjects:
            run_case_anisotropic(
                temp_organized_dir,
                subject_id, 
                target_spacing,
                patch_size,
                output_dir
            )
    else:
        # Parallel processing
        args_list = [
            (temp_organized_dir, subject_id, target_spacing, patch_size, output_dir)
            for subject_id in subjects
        ]
        
        with Pool(processes=num_workers) as pool:
            pool.starmap(run_case_anisotropic, args_list)
    
    # Step 4: Create cross-validation splits (stratified)
    print("Generating cross-validation splits...")
    splits_list = generate_crossval_split(subjects, n_splits=3)
    
    # Convert to expected format
    splits = {}
    for i, fold_data in enumerate(splits_list):
        splits[str(i)] = fold_data
    
    # Validate splits maintain class balance
    print("Cross-validation split summary:")
    for fold_idx, fold_data in splits.items():
        train_subjects = fold_data['train']
        val_subjects = fold_data['val']
        
        train_labels = [label_dict[s] for s in train_subjects]
        val_labels = [label_dict[s] for s in val_subjects]
        
        print(f"Fold {fold_idx}:")
        print(f"  Train: {len(train_subjects)} subjects, class dist: {np.bincount(train_labels)}")
        print(f"  Val: {len(val_subjects)} subjects, class dist: {np.bincount(val_labels)}")
    
    # Step 5: Save metadata
    with open(join(output_dir, 'splits.json'), 'w') as f:
        json.dump(splits, f, indent=2)
    
    with open(join(output_dir, 'labels.json'), 'w') as f:
        json.dump(label_dict, f, indent=2)
    
    # Step 6: Clean up temporary directory
    import shutil
    shutil.rmtree(temp_organized_dir)
    
    print("Preprocessing completed successfully!")
    print(f"Preprocessed data saved to: {output_dir}")
    print(f"Dataset summary:")
    print(f"  - {len(subjects)} subjects")
    print(f"  - 4 modalities per subject")
    print(f"  - Patch size: {patch_size}")
    print(f"  - Spacing: {target_spacing}")
    print(f"  - Class distribution: {dict(zip(*np.unique(list(label_dict.values()), return_counts=True)))}")


if __name__ == '__main__':
    # Configuration
    raw_data_dir = "/data/weidong/fomo_finetune/fomo-task1"
    output_dir = "/data/weidong/workspace/SSL3D_classification/data/fomo_task1_preprocessed"
    
    # Anisotropic preprocessing parameters
    # Note: spacing and patch_size are in (z, y, x) order to match numpy array indexing
    target_spacing = (5.0, 0.45, 0.45)  # (axial, sagittal, coronal)
    patch_size = (32, 256, 256)         # (depth, height, width)
    
    print(f"Target spacing (z,y,x): {target_spacing}")
    print(f"Target patch size (z,y,x): {patch_size}")
    
    # Run preprocessing
    preprocess_fomo_task1_dataset(
        raw_data_dir=raw_data_dir,
        output_dir=output_dir,
        target_spacing=target_spacing,
        patch_size=patch_size,
        num_workers=1  # Use 1 for debugging, increase for production
    )
