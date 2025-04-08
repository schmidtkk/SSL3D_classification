import os
import numpy as np
import pandas as pd
import json
from datasets.preprocess_3D_data.preprocess_dataset import preprocess_dataset
from batchgenerators.utilities.file_and_folder_operations import *


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



    # preprocess_dataset(nii_files, unique_ids, record_label_dict, 'median', out_folder='/home/c306h/cluster-data/classification/GLvsL_median_shape',  num_worker=8,)
    preprocess_dataset(nii_files, unique_ids, record_label_dict, [160,160,160], out_folder='/home/c306h/cluster-data/classification/GLvsL_fixed160patchsize',  num_worker=12)