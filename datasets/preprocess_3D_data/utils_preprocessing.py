import SimpleITK as sitk
import numpy as np
import blosc2
import os
from multiprocessing import Pool
import numpy as np
from cropping import crop_to_nonzero
from normalization import ZScoreNormalization
from default_resampling import resample_data_or_seg_to_shape
from blosc_helper import save_case, comp_blosc2_params


def run_case_npy(data: np.ndarray, original_spacing: list, target_shape:list):
    # let's not mess up the inputs!
    data = data.astype(np.float32)  # this creates a copy

    # data, seg, bbox = crop_to_nonzero(data)
    old_shape = data.shape[1:]
    # print(original_spacing, target_shape, old_shape)
    # resample
    new_spacing = np.array([i / j * k for i, j, k in zip(original_spacing, target_shape, old_shape)])

    # normalize
    # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
    # longer fitting the images perfectly!
    normalizer = ZScoreNormalization()
    data = normalizer.run(data)

    # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
    #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
    data = resample_data_or_seg_to_shape(data, target_shape, original_spacing, new_spacing)
    print(f'old shape: {old_shape}, new_shape: {target_shape}, old_spacing: {original_spacing}, '
            f'new_spacing: {new_spacing}')
    return data

def read_spacing_and_size(path):
    """Helper function to read a NIfTI file and return its spacing and size."""
    img = sitk.ReadImage(path)
    # print(path)
    return img.GetSpacing(), img.GetSize()

def get_median_spacing_and_size_distributed(nifti_filepaths, num_workers=4):
    """
    Distributed version: Computes the median spacing and size of NIfTI files using multiprocessing.

    Args:
        nifti_filepaths (list of str): List of paths to .nii.gz files.
        num_workers (int or None): Number of processes to use. Defaults to number of CPU cores.

    Returns:
        tuple:
            median_spacing (tuple of float): Median spacing (x, y, z).
            median_size (tuple of int): Median image size (x, y, z).
    """

    with Pool(num_workers) as pool:
        results = pool.map(read_spacing_and_size, nifti_filepaths)

    spacings, sizes = zip(*results)
    spacings_array = np.array(spacings)
    sizes_array = np.array(sizes)

    median_spacing = tuple(np.median(spacings_array, axis=0))
    median_size = tuple(np.median(sizes_array, axis=0).astype(int))

    return median_spacing, median_size, spacings, sizes


def run_case(file_path, target_shape, output_dir):
    img = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(img)
    data = data.reshape((1, *data.shape))
    original_spacing = img.GetSpacing()
    result = run_case_npy(data, original_spacing, target_shape)
    result = result.astype(np.float32, copy=False)
    file_name = os.path.basename(file_path)[:-7]
    out_path_truncated = os.path.join(output_dir, file_name)


    block_size_data, chunk_size_data = comp_blosc2_params(
        result.shape,
        target_shape,
        data.itemsize)

    save_case(result, out_path_truncated, chunks=chunk_size_data, blocks=block_size_data)


def run_all_cases(filepaths, target_shape, output_dir, num_workers:int=4):
    os.makedirs(output_dir, exist_ok=True)
    args_list = [(fp, target_shape, output_dir) for fp in filepaths]

    with Pool(processes=num_workers) as pool:
        pool.starmap(run_case, args_list)