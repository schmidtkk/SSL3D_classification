import shutil
import pandas as pd
from datasets.preprocess_3D_data.datasets.template_brain_preprocessing import *

from datasets.preprocess_3D_data.blosc_helper import save_case, comp_blosc2_params
from batchgenerators.utilities.file_and_folder_operations import *

from datasets.preprocess_3D_data.preprocess_dataset import preprocess_dataset_tospacing


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
    base_path = "/home/c306h/cluster-data/ssl3d_data/classification/raw/rec_vs_t/"
    out_dir = "/home/c306h/cluster-data/ssl3d_data/classification/preprocessed/rec_vs_t_1mm_cropped_160_new"
    raw_data_dir = '/home/c306h/cluster-data/ssl3d_data/classification/raw/rec_vs_t_1mm_cropped_160_new'
    maybe_mkdir_p(join(raw_data_dir, 'imagesTr'))

    # Collect matching files
    nii_files = []
    unique_ids = []

    for file in os.listdir(base_path):
        if file.endswith("MR_pseudo_ax_km.nii.gz"):
            if file[:8] not in ['subj_046', 'subj_041', 'subj_105']:
                unique_ids.append(file[:8])
                shutil.copy(join(base_path, file), join(raw_data_dir, 'imagesTr',file[:8] + '_0000.nii.gz'))
                shutil.copy(join(base_path, file[:8] + '_MR_postop_ax_km_reg.nii.gz'), join(raw_data_dir, 'imagesTr', file[:8] + '_0001.nii.gz'))



    record_label_dict = extract_label_dict_from_excel(join(base_path, 'metadata.xlsx'))
    maybe_mkdir_p(out_dir)
    save_json(record_label_dict, join(out_dir, 'labels.json'))

    # predict brainmasks
    hd_bet_predict(raw_data_dir)
    load_crop_brainextract_normalize_images(raw_data_dir, out_dir, [1.,1.,1.], [160,160,160], brain_extract=True, num_workers=5)




