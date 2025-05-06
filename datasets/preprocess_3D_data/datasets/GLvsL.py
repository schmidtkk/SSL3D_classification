import os
import numpy as np
import pandas as pd
import json

from datasets.preprocess_3D_data.crop_to_mask import process_and_save_all_cases
from datasets.preprocess_3D_data.preprocess_dataset import preprocess_dataset_toshape, preprocess_dataset_tospacing
from batchgenerators.utilities.file_and_folder_operations import *


if __name__ == '__main__':
    # Base path and folders to search
    base_path = ""  # Change this to your actual base directory

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


    #resampling to 1mm spacing
    preprocess_dataset_tospacing(nii_files, unique_ids, record_label_dict, [1.,1.,1.], out_folder='',  num_worker=12)

    ###########use hdbet to find brain center ##########
    ###########hd-bet -i imagepath -o outpath --save_bet_mask --no_bet_image######


    # image_dir = "imagepath"
    # mask_dir = "finalpath"
    # out_dir = "outpath"
    #
    # process_and_save_all_cases(
    #     image_dir=image_dir,
    #     mask_dir=mask_dir,
    #     out_dir=out_dir,
    #     target_shape=(160, 160, 160),
    #     num_workers=12)  # Adjust for your system



