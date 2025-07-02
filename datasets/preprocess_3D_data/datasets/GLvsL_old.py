import shutil

from datasets.preprocess_3D_data.crop_to_mask import load_image_np, crop_center_with_padding_np, get_mask_center
import glob
import numpy as np
from multiprocessing import Pool
from functools import partial

from datasets.preprocess_3D_data.blosc_helper import save_case, comp_blosc2_params
from batchgenerators.utilities.file_and_folder_operations import *

from datasets.preprocess_3D_data.preprocess_dataset import preprocess_dataset_tospacing


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

if __name__ == '__main__':
    # Base path and folders to search
    base_path = "/home/c306h/E132-Projekte/Projects/2022_Miriam_UKHD_NeuroSurgery_GlioVsLym/Classification_SSL3D/"  # Change this to your actual base directory
    # Collect matching files
    nii_files = []
    file_names = []

    for folder in os.listdir(base_path):
        file_names.append(folder)
        for file in listdir(join(base_path, folder)):
            if file.endswith("reg.nii.gz") or file.startswith("t1_"):
                nii_files.append(join(base_path, folder, file))


    unique_ids = np.unique(file_names)
    print(f'Unique IDs: {unique_ids}', len(unique_ids))
    record_label_dict = {}
    for case in unique_ids:
        if case.startswith('G'):
            record_label_dict[case] = 1
        else:
            record_label_dict[case] = 0


    #resampling to 1mm spacing - uncomment!!
    # preprocess_dataset_tospacing(nii_files, unique_ids, record_label_dict, [1.,1.,1.], out_folder='/home/c306h/cluster-data/ssl3d_data/classification/raw/GLvsL_1mm',  num_worker=12)

    ###########use hdbet to find brain center ##########
    ###########hd-bet -i imagepath -o outpath --save_bet_mask --no_bet_image######

    #
    image_dir = '/home/c306h/cluster-data/ssl3d_data/classification/raw/GLvsL_1mm'
    mask_dir = '/home/c306h/cluster-data/ssl3d_data/classification/raw/GLvsL_1mm/masks'
    out_dir = "/home/c306h/cluster-data/ssl3d_data/classification/preprocessed/GLvsL_1mm_cropped_160"
    maybe_mkdir_p(out_dir)
    shutil.copy(join(image_dir, 'labels.json'), join(out_dir, 'labels.json'))
    shutil.copy(join(image_dir, 'splits.json'), join(out_dir, 'splits.json'))

    process_and_save_all_cases(
        image_dir=image_dir,
        mask_dir=mask_dir,
        out_dir=out_dir,
        target_shape=(160, 160, 160), #we use a 160pix patchsize for the pre-training
        num_workers=12)  # Adjust for your system



