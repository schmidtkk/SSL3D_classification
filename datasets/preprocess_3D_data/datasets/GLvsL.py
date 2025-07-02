import shutil
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from datasets.preprocess_3D_data.datasets.template_brain_preprocessing import *

if __name__ == '__main__':
    # Base path and folders to search
    base_path = "/home/c306h/E132-Projekte/Projects/2022_Miriam_UKHD_NeuroSurgery_GlioVsLym/Classification_SSL3D/"
    out_dir = "/home/c306h/cluster-data/ssl3d_data/classification/preprocessed/GLvsL_1mm_cropped_160_new"
    raw_data_dir = '/home/c306h/cluster-data/ssl3d_data/classification/raw/GLvsL_1mm_cropped_160_new'
    maybe_mkdir_p(join(raw_data_dir, 'imagesTr'))

    # Collect matching files
    nii_files = []
    file_names = []

    for folder in os.listdir(base_path):
        file_names.append(folder)
        for file in listdir(join(base_path, folder)):
            if file.endswith("reg.nii.gz") or file.startswith("t1_"):
                if file.startswith("t1_"):
                    ending = '_0000.nii.gz'
                elif file.startswith("t2_"):
                    ending = '_0001.nii.gz'
                elif file.startswith("t1ce_"):
                    ending = '_0002.nii.gz'
                elif file.startswith("flair_"):
                    ending = '_0003.nii.gz'
                nii_files.append(join(base_path, folder, file))
                shutil.copy(join(base_path, folder, file), join(raw_data_dir, 'imagesTr', folder + ending))


    unique_ids = np.unique(file_names)
    print(f'Unique IDs: {unique_ids}', len(unique_ids))
    record_label_dict = {}
    for case in unique_ids:
        if case.startswith('G'):
            record_label_dict[case] = 1
        else:
            record_label_dict[case] = 0

    maybe_mkdir_p(out_dir)
    save_json(record_label_dict, join(out_dir, 'labels.json'))

    # predict brainmasks
    hd_bet_predict(raw_data_dir)
    load_crop_brainextract_normalize_images(raw_data_dir, out_dir, [1.,1.,1.], [160,160,160], brain_extract=True, num_workers=5)


