import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from datasets.preprocess_3D_data.hd_bet_prediction import hdbet_predict, get_hdbet_predictor
from datasets.preprocess_3D_data.crop_to_mask import load_image_np, crop_center_with_padding_np, get_mask_center
from datasets.preprocess_3D_data.default_resampling import resample_data_or_seg_to_spacing, resample_data_or_seg_to_shape
import numpy as np
from datasets.preprocess_3D_data.cross_validation import generate_crossval_split
from multiprocessing import Pool
import SimpleITK as sitk
from datasets.preprocess_3D_data.blosc_helper import save_case, comp_blosc2_params
from batchgenerators.utilities.file_and_folder_operations import *
from datasets.preprocess_3D_data.normalization import ZScoreNormalization
import re
from collections import defaultdict

def hd_bet_predict(in_folder:str):
    #predicts only mask for image with _0000.nii.gz
    predictor = get_hdbet_predictor(in_folder)
    hdbet_predict(join(in_folder, 'imagesTr'), join(in_folder, 'masks'), predictor, keep_brain_mask=True, compute_brain_extracted_image=False)


def run_case(base_path, img_id, target_spacing, crop_size, output_dir, num_modalities, brain_extract=True):
    bas_img_path = join(base_path, 'imagesTr', img_id)
    if isfile(join(base_path, 'masks', img_id + '_0000_bet.nii.gz')):
        has_mask = True
        mask_path = join(base_path, 'masks', img_id + '_0000_bet.nii.gz')
    else:
        has_mask = False

    if has_mask:
        mask_img = sitk.ReadImage(mask_path)
        mask_data = sitk.GetArrayFromImage(mask_img)
        mask_data = mask_data.reshape(1, mask_data.shape[0], mask_data.shape[1], mask_data.shape[2])
        original_spacing = mask_img.GetSpacing()[::-1]
        mask_1mm = resample_data_or_seg_to_spacing(mask_data, original_spacing, target_spacing, is_seg=True)
        center = get_mask_center(mask_1mm[0])
        resized_mask = crop_center_with_padding_np(mask_1mm[0], center, crop_size)


    for mod_id in range(num_modalities):
        img = sitk.ReadImage(join(bas_img_path + f'_000{mod_id}.nii.gz'))
        data = sitk.GetArrayFromImage(img)
        data = data.reshape(1, data.shape[0], data.shape[1], data.shape[2])

        #resample image
        data_1mm = resample_data_or_seg_to_spacing(data, original_spacing, target_spacing)

        if has_mask:
            if brain_extract:
                data_1mm = data_1mm * mask_1mm

            # crop around mask center
            resized_img = crop_center_with_padding_np(data_1mm[0], center, crop_size)
            resized_img = resized_img.reshape(1, resized_img.shape[0], resized_img.shape[1], resized_img.shape[2])
        else:
            new_spacing = np.array([i / j * k for i, j, k in zip(target_spacing, crop_size, data_1mm.shape[1:])])
            resized_img = resample_data_or_seg_to_shape(data_1mm, crop_size, target_spacing, new_spacing)

        #####normalize
        normalizer = ZScoreNormalization()
        if has_mask:
            normalizer.use_mask_for_norm = True
            data = normalizer.run(resized_img, resized_mask.reshape(1, resized_mask.shape[0], resized_mask.shape[1], resized_mask.shape[2]))
        else:
            data = normalizer.run(resized_img)


        #save_blosc_img
        block_size_data, chunk_size_data = comp_blosc2_params(resized_img.shape, crop_size ,data.itemsize)
        out_path_truncated = join(output_dir, f'{img_id}_000{mod_id}')
        save_case(data, out_path_truncated, chunks=chunk_size_data, blocks=block_size_data)



def load_crop_brainextract_normalize_images(in_folder:str , out_folder:str, target_spacing:list, patch_size:list, brain_extract=True, num_workers:int=1 ):
    os.makedirs(out_folder, exist_ok=True)
    all_images = os.listdir(join(in_folder, 'imagesTr'))
    unique_ids = list(set(re.sub(r'_\d{4}\.nii\.gz$', '', f) for f in all_images))

    pattern = re.compile(r'(.+?)_(\d{4})\.nii\.gz')
    modalities = defaultdict(set)

    for f in all_images:
        m = pattern.match(f)
        if m:
            identifier, mod = m.groups()
            modalities[identifier].add(mod)

    expected = set.union(*modalities.values())
    incomplete = {k: sorted(expected - v) for k, v in modalities.items() if v != expected}

    if incomplete:
        msg = '\n'.join(f"{k} missing: {v}" for k, v in incomplete.items())
        raise RuntimeError(f"Some identifiers are missing modalities:\n{msg}")


    #preprocess cases: Original image → resampling to 1mm → cropping to [160,160,160] using center of mask or resizing whole image if no mask available → maybe skull-stripping → z-score normalization maybe only for brain mask
    args_list = [(in_folder, id,  target_spacing, patch_size, out_folder, len(expected), brain_extract) for id in unique_ids]

    with Pool(processes=num_workers) as pool:
        pool.starmap(run_case, args_list)



    split_file = generate_crossval_split(unique_ids, n_splits=3)

    with open(os.path.join(out_folder, 'splits.json'), "w") as f:
        json.dump(split_file, f, indent=2)



def create_label_dict(path):
    label_dict = load_json(path)
    return label_dict

if __name__ == '__main__':
    '''
    1. organize the raw data like nnU-Net: raw folder has a lable dict (# {'unique_id1': 1, ...} and imagesTr with all images files
    case_identifier_0000.nii.gz and _000x.nii.gz for other modallities
    2. potentially use HD-Bet for brain extraction - saved in masks folder (only applied to modallity 0)
    3. resample mask and image to 1mm spacing
    4. crop image to center of mask with a fixed FOV patch size
    5. z-score normalization of resulting crop
    '''
