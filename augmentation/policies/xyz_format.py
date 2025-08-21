"""
XYZ Format Augmentation Policies (NNSSL Compatible)
Designed for data in (C, X, Y, Z) format where Z is the anisotropic dimension
"""

import numpy as np
from batchgenerators.transforms.spatial_transforms import (
    MirrorTransform,
    SpatialTransform
)
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    GammaTransform
)
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor

def get_fomo_task1_xyz_training_transforms(
    patch_size=(256, 256, 32),  # XYZ format
    rotation_for_DA: float = 0.523599,  # ~30 degrees
    mirror_axes: tuple = (0, 1),  # Mirror X and Y axes only (preserve anisotropic Z)
    **kwargs
):
    """
    Create training transforms for FOMO Task1 data in XYZ format (NNSSL compatible)
    
    Args:
        patch_size: Target patch size in XYZ format (256, 256, 32)
        rotation_for_DA: Rotation angle in radians (~30 degrees default)
        mirror_axes: Axes to mirror (0, 1) = (X, Y) - don't mirror Z axis
        
    Returns:
        Composed transform pipeline
    """
    
    print("XYZ augmentation pipeline created:")
    print(f"  Patch size: {patch_size}")
    print(f"  Mirror axes: {mirror_axes} (X, Y only - preserving anisotropic Z)")
    print(f"  Rotation range: ±{rotation_for_DA * 180 / np.pi:.1f}°")
    print("  Number of transforms: 7")
    
    transforms = []
    
    # 1. Mirror transform - only in XY plane to preserve anisotropic Z
    transforms.append(
        MirrorTransform(
            axes=mirror_axes,  # (0, 1) = (X, Y) in XYZ format
            p_per_sample=0.5
        )
    )
    
    # 2. Spatial transforms - conservative for anisotropic data
    transforms.append(
        SpatialTransform(
            patch_size=patch_size,
            patch_center_dist_from_border=patch_size[2] // 2,  # Use Z dimension for border
            do_elastic_deform=True,
            alpha=(0., 200.),  # Reduced for medical data
            sigma=(9., 13.),
            do_rotation=True,
            angle_x=(-rotation_for_DA, rotation_for_DA),  # Tuple format required
            angle_y=(-rotation_for_DA, rotation_for_DA),  # Tuple format required
            angle_z=(-rotation_for_DA * 0.5, rotation_for_DA * 0.5),  # Tuple format required
            do_scale=True,
            scale=(0.85, 1.15),  # Conservative scaling
            border_mode_data='constant',
            border_cval_data=0,
            order_data=3,  # Bicubic interpolation
            random_crop=False,  # Already have target size
            p_el_per_sample=0.3,  # Reduced elastic deformation probability
            p_scale_per_sample=0.3,
            p_rot_per_sample=0.3
        )
    )
    
    # 3. Brightness multiplicative
    transforms.append(
        BrightnessMultiplicativeTransform(
            multiplier_range=(0.8, 1.2),
            p_per_sample=0.3
        )
    )
    
    # 4. Gamma transform
    transforms.append(
        GammaTransform(
            gamma_range=(0.8, 1.2),
            invert_image=False,
            p_per_sample=0.3,
            retain_stats=True
        )
    )
    
    # 5. Gaussian noise
    transforms.append(
        GaussianNoiseTransform(
            noise_variance=(0, 0.05),
            p_per_sample=0.3
        )
    )
    
    return Compose(transforms)

def get_fomo_task1_xyz_validation_transforms(
    patch_size=(256, 256, 32),  # XYZ format
    **kwargs
):
    """
    Create validation transforms (minimal) for XYZ format
    """
    # No augmentation for validation
    return Compose([])

class XYZFormatAdapter:
    """
    Adapter to ensure transforms work correctly with XYZ data format
    Input: (C, X, Y, Z) tensor
    Output: (C, X, Y, Z) tensor
    """
    
    def __init__(self, base_transform):
        self.base_transform = base_transform
    
    def __call__(self, data):
        """
        Apply transforms to XYZ format data
        
        Args:
            data: numpy array of shape (C, X, Y, Z)
            
        Returns:
            transformed data of shape (C, X, Y, Z)
        """
        
        # Verify input format
        assert len(data.shape) == 4, f"Expected 4D input (C, X, Y, Z), got {data.shape}"
        assert data.shape[-1] <= data.shape[-2], f"Z dimension should be smallest for anisotropic data"
        
        # Apply transforms
        # Note: batchgenerators expects data as dict
        data_dict = {'data': data[None, ...]}  # Add batch dimension: (1, C, X, Y, Z)
        
        transformed = self.base_transform(**data_dict)
        
        # Remove batch dimension and return
        return transformed['data'][0]  # Back to (C, X, Y, Z)

def create_xyz_compatible_transforms(patch_size=(256, 256, 32), **kwargs):
    """
    Create transform pipeline that's explicitly compatible with XYZ format
    """
    base_transforms = get_fomo_task1_xyz_training_transforms(
        patch_size=patch_size, 
        **kwargs
    )
    
    return XYZFormatAdapter(base_transforms)

# Convenience functions for config files
def get_fomo_task1_xyz_transforms(patch_size, **kwargs):
    """Entry point for config files - XYZ format"""
    return get_fomo_task1_xyz_training_transforms(
        patch_size=tuple(patch_size), 
        **kwargs
    )

if __name__ == "__main__":
    # Test the transforms
    print("Testing XYZ format transforms...")
    
    # Create dummy data in XYZ format
    dummy_data = np.random.randn(4, 256, 256, 32).astype(np.float32)
    print(f"Input shape: {dummy_data.shape} (C, X, Y, Z)")
    
    # Create transforms
    transforms = create_xyz_compatible_transforms()
    
    # Apply transforms
    transformed = transforms(dummy_data)
    print(f"Output shape: {transformed.shape} (C, X, Y, Z)")
    
    # Verify shape is preserved
    assert transformed.shape == dummy_data.shape, "Shape should be preserved"
    print("✅ XYZ transforms working correctly!")
