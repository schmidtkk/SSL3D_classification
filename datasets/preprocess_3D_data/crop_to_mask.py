import os
import glob
import numpy as np
import SimpleITK as sitk
from multiprocessing import Pool
from functools import partial

from datasets.preprocess_3D_data.blosc_helper import save_case, comp_blosc2_params


def load_image_np(path):
    """Load NIfTI image and return NumPy array and reference image for metadata."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    return arr, img


def get_mask_center(mask_arr):
    """Return (z, y, x) center of non-zero mask voxels."""
    coords = np.argwhere(mask_arr == 1)
    if coords.size == 0:
        raise ValueError("Mask is empty (no voxels with value 1).")
    return tuple(coords.mean(axis=0).astype(int))


def crop_center_with_padding_np(arr, center, size=(160, 160, 160)):
    """Crop a 3D NumPy array with zero padding around a center (z, y, x)."""
    zc, yc, xc = center
    dz, dy, dx = size[0] // 2, size[1] // 2, size[2] // 2

    z_start, z_end = zc - dz, zc + dz
    y_start, y_end = yc - dy, yc + dy
    x_start, x_end = xc - dx, xc + dx

    cropped = np.zeros(size, dtype=arr.dtype)

    z_start_valid = max(z_start, 0)
    y_start_valid = max(y_start, 0)
    x_start_valid = max(x_start, 0)

    z_end_valid = min(z_end, arr.shape[0])
    y_end_valid = min(y_end, arr.shape[1])
    x_end_valid = min(x_end, arr.shape[2])

    z_off = z_start_valid - z_start
    y_off = y_start_valid - y_start
    x_off = x_start_valid - x_start

    cropped[
    z_off:z_off + (z_end_valid - z_start_valid),
    y_off:y_off + (y_end_valid - y_start_valid),
    x_off:x_off + (x_end_valid - x_start_valid)
    ] = arr[
        z_start_valid:z_end_valid,
        y_start_valid:y_end_valid,
        x_start_valid:x_end_valid
        ]

    return cropped


def load_and_crop_modalities_by_mask_center(
        image_id,
        image_dir,
        mask_dir,
        crop_size=(160, 160, 160)
):
    """Load 4 modalities + mask, crop around mask center, return dict of arrays."""
    t1, _ = load_image_np(os.path.join(image_dir, f"t1_img_{image_id}.nii.gz"))
    t2, _ = load_image_np(os.path.join(image_dir, f"t2_img_{image_id}_reg.nii.gz"))
    t1ce, _ = load_image_np(os.path.join(image_dir, f"t1ce_img_{image_id}_reg.nii.gz"))
    flair, _ = load_image_np(os.path.join(image_dir, f"flair_img_{image_id}_reg.nii.gz"))
    mask, _ = load_image_np(os.path.join(mask_dir, f"t1_img_{image_id}_bet.nii.gz"))

    center = get_mask_center(mask)

    return {
        "t1": crop_center_with_padding_np(t1, center, crop_size),
        "t2": crop_center_with_padding_np(t2, center, crop_size),
        "t1ce": crop_center_with_padding_np(t1ce, center, crop_size),
        "flair": crop_center_with_padding_np(flair, center, crop_size),
    }


def process_single_case(image_id, image_dir, mask_dir, out_dir, crop_size):
    try:
        modalities = load_and_crop_modalities_by_mask_center(
            image_id=image_id,
            image_dir=image_dir,
            mask_dir=mask_dir,
            crop_size=crop_size
        )
        for modality_name, arr in modalities.items():
            expanded_arr = np.expand_dims(arr, axis=0)
            # print(arr.shape, crop_size, arr.itemsize, expanded_arr.shape)
            blocks, chunks = comp_blosc2_params(
                expanded_arr.shape, crop_size, arr.itemsize
            )

            out_path_truncated = os.path.join(
                out_dir, f"{modality_name}_img_{image_id}"
            )

            save_case(expanded_arr, out_path_truncated, chunks=chunks, blocks=blocks)

            print(f"✅ Saved cropped data for image {image_id}")
    except Exception as e:
        print(f"❌ Failed to process {image_id}: {e}")


def process_and_save_all_cases(
        image_dir,
        mask_dir,
        out_dir,
        target_shape=(160, 160, 160),
        num_workers=8
):
    os.makedirs(out_dir, exist_ok=True)

    t1_paths = sorted(glob.glob(os.path.join(image_dir, "t1_img_*.nii.gz")))
    image_ids = [
        os.path.basename(p).replace("t1_img_", "").replace(".nii.gz", "")
        for p in t1_paths
    ]

    with Pool(processes=num_workers) as pool:
        pool.map(
            partial(
                process_single_case,
                image_dir=image_dir,
                mask_dir=mask_dir,
                out_dir=out_dir,
                crop_size=target_shape
            ),
            image_ids
        )