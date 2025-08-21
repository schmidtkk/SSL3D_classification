"""
Anisotropic augmentation policies for medical data with different axial and in-plane resolutions
Specifically designed for (32, 256, 256) patches with (5.0, 0.45, 0.45) spacing
"""

from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import (
    MultiplicativeBrightnessTransform,
)
from batchgeneratorsv2.transforms.intensity.contrast import (
    ContrastTransform,
    BGContrast,
)
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import (
    SimulateLowResolutionTransform,
)
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from typing import Tuple, Union, List
import numpy as np


def get_anisotropic_training_transforms(
    patch_size: Union[np.ndarray, Tuple[int]] = (32, 256, 256),
    rotation_for_DA: RandomScalar = 0.523599,  # ~30 degrees
    mirror_axes: Tuple[int, ...] = (1, 2),  # Only mirror in-plane (YX), not axial (Z)
    do_dummy_2d_data_aug: bool = False,
) -> BasicTransform:
    """
    Get training transforms optimized for anisotropic medical data.
    
    Args:
        patch_size: Target patch size (Z, Y, X) = (32, 256, 256)
        rotation_for_DA: Maximum rotation angle for in-plane rotations
        mirror_axes: Axes to mirror - (1,2) for YX only, avoid axial mirroring
        do_dummy_2d_data_aug: If True, treat as 2D slices (not recommended for 3D data)
        
    Returns:
        Composed transform pipeline
    """
    transforms = []
    
    if do_dummy_2d_data_aug:
        print("Warning: Using 2D augmentations on 3D data - may not be optimal")
        # For anisotropic data, we might want slice-wise augmentations
        ignore_axes = (0,)  # Ignore Z axis
        patch_size_spatial = patch_size[1:]  # (Y, X)
    else:
        # Full 3D augmentations with anisotropic awareness
        patch_size_spatial = patch_size
        ignore_axes = None
    
    # Spatial transforms with anisotropic considerations
    transforms.append(
        SpatialTransform(
            patch_size_spatial,
            patch_center_dist_from_border=0,
            random_crop=False,
            # Rotation: primarily in-plane (XY), limited around Z-axis
            p_rotation=0.3,
            rotation=rotation_for_DA,  # Only around Z-axis for anisotropic data
            # Scaling: conservative to preserve anatomical relationships
            p_scaling=0.2,
            scaling=(0.8, 1.25),  # More conservative scaling
            p_synchronize_scaling_across_axes=1,
            # Elastic deformation: limited for anisotropic data
            p_elastic_deform=0.1,  # Reduced probability
            bg_style_seg_sampling=False,
        )
    )
    
    # Intensity augmentations (safe for anisotropic data)
    transforms.append(
        RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1), 
                p_per_channel=1, 
                synchronize_channels=False  # Different modalities may need different noise
            ),
            apply_probability=0.15,
        )
    )
    
    # Gaussian blur: different sigma for different axes due to anisotropy
    transforms.append(
        RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.3, 0.8),  # Smaller sigma range for anisotropic data
                synchronize_channels=False,
                synchronize_axes=False,  # Different blur for different axes
                p_per_channel=0.3,  # Lower probability per channel
                benchmark=True,
            ),
            apply_probability=0.15,
        )
    )
    
    # Brightness augmentation: per-channel for multi-modal data
    transforms.append(
        RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.8, 1.2)),  # Conservative range
                synchronize_channels=False,  # Important for multi-modal
                p_per_channel=0.8,
            ),
            apply_probability=0.2,
        )
    )
    
    # Contrast augmentation: per-channel for multi-modal data  
    transforms.append(
        RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.8, 1.2)),  # Conservative range
                preserve_range=True,
                synchronize_channels=False,  # Important for multi-modal
                p_per_channel=0.8,
            ),
            apply_probability=0.2,
        )
    )
    
    # Low resolution simulation: careful with anisotropic data
    transforms.append(
        RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.7, 1),  # Conservative scaling
                synchronize_channels=False,
                synchronize_axes=False,  # Different scaling per axis
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.3,
            ),
            apply_probability=0.2,
        )
    )
    
    # Gamma augmentation: conservative for medical data
    transforms.append(
        RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.8, 1.3)),  # Conservative gamma range
                p_invert_image=0.1,  # Low inversion probability
                synchronize_channels=False,
                p_per_channel=0.5,
                p_retain_stats=1,
            ),
            apply_probability=0.15,
        )
    )
    
    # Second gamma transform without inversion
    transforms.append(
        RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.8, 1.3)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=0.5,
                p_retain_stats=1,
            ),
            apply_probability=0.25,
        )
    )
    
    # Mirroring: only in-plane for anisotropic data
    if mirror_axes is not None and len(mirror_axes) > 0:
        print(f"Mirroring enabled for axes: {mirror_axes}")
        transforms.append(MirrorTransform(allowed_axes=mirror_axes))
    
    print("Anisotropic augmentation pipeline created:")
    print(f"  Patch size: {patch_size}")
    print(f"  Mirror axes: {mirror_axes}")
    print(f"  Rotation range: ±{np.degrees(rotation_for_DA):.1f}°")
    print(f"  Number of transforms: {len(transforms)}")
    
    return ComposeTransforms(transforms)


def get_anisotropic_validation_transforms() -> BasicTransform:
    """
    Get validation transforms for anisotropic data.
    Minimal transforms to avoid changing the data during validation.
    """
    transforms = []
    
    # For validation, we typically don't apply augmentations
    # Just ensure data format is correct
    print("Anisotropic validation transforms: no augmentations")
    
    return ComposeTransforms(transforms)


# Convenience function for integration
def get_fomo_task1_training_transforms(
    patch_size: Tuple[int, int, int] = (32, 256, 256),
    rotation_for_DA: float = 0.523599,
    mirror_axes: Tuple[int, ...] = (1, 2),
) -> BasicTransform:
    """
    Get training transforms specifically configured for FOMO Task1 dataset.
    
    Args:
        patch_size: (32, 256, 256) for anisotropic patches
        rotation_for_DA: ~30 degrees in radians
        mirror_axes: (1, 2) to mirror only in YX plane
    """
    return get_anisotropic_training_transforms(
        patch_size=patch_size,
        rotation_for_DA=rotation_for_DA,
        mirror_axes=mirror_axes,
        do_dummy_2d_data_aug=False,
    )
