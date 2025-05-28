import os
import shutil

import pandas as pd
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

    img_paths = [i for i in os.listdir(image_dir) if i.endswith('.nii.gz')]
    image_ids = [p.split('/')[-1][:-7] for p in img_paths]

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
                out_dir, f"{image_id}_{modality_name}"
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
    img, _ = load_image_np(os.path.join(image_dir, f"{image_id}.nii.gz"))
    mask, _ = load_image_np(os.path.join(mask_dir , f"{image_id}_bet.nii.gz"))

    center = get_mask_center(mask)

    return {
        "crop": crop_center_with_padding_np(img, center, crop_size),
    }

def create_label_dict(path):
    label_dict = load_json(path)
    return label_dict

if __name__ == '__main__':
    # Base path and folders to search
    '''
    Steps to implement yourself: 
    1. get all nifti files (nii_files)
    2. get all case identifier (unique ids)
    3. create a lable dict containing class labels 
    4. use preprocess_dataset_tospacing function to resample all images to 1mm spacing
    5. apply hdbet to all image to find brain center (https://github.com/MIC-DKFZ/HD-BET)
    6. copy labelsjson and splits.json file to preprocessed folder
    6. crop all images given the HDbet masks and save them as blosc  in preprocessed folder for training ( process_and_save_all_cases needs to be adapted for your dataset)
    
    
    '''
    base_dir = '/home/c306h/cluster-data_all/t006d/Datasets/ABIDE/ABIDE_img'
    csv_path = '/home/c306h/cluster-data_all/t006d/Datasets/ABIDE/ABIDE_2_20_2025.csv'
    # 1. Find all .nii files under MP-RAGE
    nii_files = glob.glob(os.path.join(base_dir, '**/MP-RAGE/**/*.nii'), recursive=True)

    # 2. get all case identifier (unique ids)
    unique_ids = [p.split('/') [-1] for p in nii_files]  # image fnames

    # 3. Load label CSV
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # clean up header whitespace
    df['Label'] = df['Group'].apply(lambda x: 1 if str(x).strip() == 'Autism' else 0)

    # 4. Map Subject ID to Label
    label_dict_full = dict(zip(df['Subject'].astype(str), df['Label']))

    # 5. Build label dict with full path as key
    label_dict = {}
    for full_file_name in unique_ids:
        subject_id = full_file_name.split('_')[1]
        label_dict[full_file_name] = label_dict_full.get(subject_id, None)



    # 4. use preprocess_dataset_tospacing function to resample all images to 1mm spacing
    ###############resampling to 1mm spacing - uncomment!######################
    # preprocess_dataset_tospacing(nii_files, unique_ids, label_dict, [1.,1.,1.], out_folder='/home/c306h/cluster-data/ssl3d_data/classification/raw/abide',  num_worker=12)

    ###########use hdbet to find brain center ##########
    ###########hd-bet -i imagepath -o outpath --save_bet_mask --no_bet_image######

    ###########last step: copy files, crop and save images#####################
    image_dir = '/home/c306h/cluster-data/ssl3d_data/classification/raw/abide'
    mask_dir = '/home/c306h/cluster-data/ssl3d_data/classification/raw/abide/masks'
    out_dir = "/home/c306h/cluster-data/ssl3d_data/classification/preprocessed/abide_1mm_cropped_160"
    maybe_mkdir_p(out_dir)
    shutil.copy(join(image_dir, 'labels.json'), join(out_dir, 'labels.json'))
    shutil.copy(join(image_dir, 'splits.json'), join(out_dir, 'splits.json'))

    process_and_save_all_cases(
        image_dir=image_dir,
        mask_dir=mask_dir,
        out_dir=out_dir,
        target_shape=(160, 160, 160), #we use a 160pix patchsize for the pre-training
        num_workers=12)  # Adjust for your system