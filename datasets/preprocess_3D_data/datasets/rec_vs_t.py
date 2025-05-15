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

    t1_paths = sorted(glob.glob(os.path.join(image_dir, "*_MR_pseudo_ax_km.nii.gz")))
    image_ids = [p.split('/')[-1][:8] for p in t1_paths]

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
    pseudo, _ = load_image_np(os.path.join(image_dir, f"{image_id}_MR_pseudo_ax_km.nii.gz"))
    postop, _ = load_image_np(os.path.join(image_dir, f"{image_id}_MR_postop_ax_km_reg.nii.gz"))
    mask, _ = load_image_np(os.path.join(mask_dir, f"{image_id}_MR_pseudo_ax_km_bet.nii.gz"))

    center = get_mask_center(mask)

    return {
        "MR_pseudo_ax_km": crop_center_with_padding_np(pseudo, center, crop_size),
        "MR_postop_ax_km_reg": crop_center_with_padding_np(postop, center, crop_size),
    }
def extract_label_dict_from_excel(xlsx_path):
    # Use second row as header
    df = pd.read_excel(xlsx_path, header=1)
    label_dict = {}

    for _, row in df.iterrows():
        pat_id = row['PatID']
        key = f"subj_{int(pat_id):03d}"

        try:
            rezidiv_op = int(row['Pathobefund (0=RN eindeutig, 1=Rezidiv eindeutig, 2=Mischbild, 3=uneindeutig)'])
        except (ValueError, KeyError, TypeError):
            rezidiv_op = None

        try:
            radionekrose = int(row['Radionekrose (0=nein, 1=ja, 2=Mischbild)'])
        except (ValueError, KeyError, TypeError):
            radionekrose = None

        if rezidiv_op == 1 and radionekrose != 1:
            label_dict[key] = 1
        elif radionekrose == 1 and rezidiv_op != 1:
            label_dict[key] = 0
        else:
            print(f"Warning: Unclear label for ID {key} (Rezidiv-OP: {rezidiv_op}, Radionekrose: {radionekrose})")

    return label_dict

if __name__ == '__main__':
    # Base path and folders to search
    base_path = "/home/c306h/cluster-data/ssl3d_data/classification/raw/rec_vs_t/"  # Change this to your actual base directory

    # Collect matching files
    nii_files = []
    unique_ids = []

    for file in os.listdir(base_path):
        if file.endswith("MR_pseudo_ax_km.nii.gz"):
            if file[:8] not in ['subj_046', 'subj_041', 'subj_105']:
                unique_ids.append(file[:8])
                print(file[:8])
                nii_files.append(os.path.join(base_path, file))
                nii_files.append(os.path.join(base_path, file[:8] + '_MR_postop_ax_km_reg.nii.gz'))



    record_label_dict = extract_label_dict_from_excel(join(base_path, 'metadata.xlsx'))
    print(record_label_dict)


    #resampling to 1mm spacing - uncomment!
    # preprocess_dataset_tospacing(nii_files, unique_ids, record_label_dict, [1.,1.,1.], out_folder='/home/c306h/cluster-data/ssl3d_data/classification/raw/rec_vs_t_1mm',  num_worker=12)

    ###########use hdbet to find brain center ##########
    ###########hd-bet -i imagepath -o outpath --save_bet_mask --no_bet_image######


    image_dir = '/home/c306h/cluster-data/ssl3d_data/classification/raw/rec_vs_t_1mm'
    mask_dir = "/home/c306h/cluster-data/ssl3d_data/classification/raw/rec_vs_t_1mm/masks"
    out_dir = "/home/c306h/cluster-data/ssl3d_data/classification/preprocessed/rec_vs_t_1mm_cropped_160"
    maybe_mkdir_p(out_dir)
    shutil.copy(join(image_dir, 'labels.json'), join(out_dir, 'labels.json'))
    shutil.copy(join(image_dir, 'splits.json'), join(out_dir, 'splits.json'))

    process_and_save_all_cases(
        image_dir=image_dir,
        mask_dir=mask_dir,
        out_dir=out_dir,
        target_shape=(160, 160, 160), #we use a 160pix patchsize for the pre-training
        num_workers=12)  # Adjust for your system