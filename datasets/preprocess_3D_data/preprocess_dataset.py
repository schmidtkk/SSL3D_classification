import os
import numpy as np
import pandas as pd
import json
from utils_preprocessing import get_median_spacing_and_size_distributed, run_all_cases
from cross_validation import generate_crossval_split

def preprocess_dataset(
        nii_files: list,
        unique_identifiers: list,
        label_dict: dict,
        target_img_size,  # Union[str, Tuple[int, int, int]] – left untyped for flexibility
        num_worker: int,
        out_folder: str
):
    """
    Preprocess a list of .nii.gz medical image files by computing median spacing/size
    and optionally resampling to a target size. Also saves the label dictionary.

    Args:
        nii_files (List[str]):
            A list of paths to .nii.gz image files.

        unique_identifiers (List[str]):
            A list of unique identifiers corresponding to each case.

        label_dict (Dict[str, int]):
            A mapping from class labels (as strings) to integers.

        target_img_size (Union[str, Tuple[int, int, int]]):
            Target image shape. Can be a tuple (e.g., (128, 128, 128))
            or the string 'median' to use median shape.

        num_worker (int):
            Number of worker processes for parallel computation.

        out_folder (str):
            Output directory to save processed data and the labels.json file.
    """
    print(f"Total .nii.gz files found: {len(nii_files)}")

    median_spacing, median_size, all_spacings, all_shapes = get_median_spacing_and_size_distributed(
        nii_files, num_workers=num_worker
    )

    print('median shape:', median_size)

    if type(target_img_size) == str:
        if target_img_size == 'median':
            run_all_cases(nii_files, median_size[::-1], out_folder, num_workers=num_worker)
    else:
        run_all_cases(nii_files, target_img_size, out_folder, num_workers=num_worker)

    with open(os.path.join(out_folder, 'labels.json'), "w") as f:
        json.dump(label_dict, f, indent=2)


    split_file = generate_crossval_split(unique_ids, n_splits=3)

    with open(os.path.join(out_folder, 'splits.json'), "w") as f:
        json.dump(split_file, f, indent=2)



if __name__ == '__main__':
    # Base path and folders to search
    base_path = "/home/c306h/E132-Projekte/Projects/2022_Peretzke_Interactive_Fiber_Dissection/data/WhiteCat/WhiteCAT_FA_MD_consti/"  # Change this to your actual base directory

    # Collect matching files
    nii_files = []
    file_names = []

    for file in os.listdir(base_path):
        if file.endswith(".nii.gz"):
            nii_files.append(os.path.join(base_path, file))
            file_names.append(file[:-10])

    unique_ids = np.unique(file_names)
    print(f'Unique IDs: {unique_ids}', len(unique_ids))


    # generat label_json
    df = pd.read_excel("/home/c306h/E132-Projekte/Projects/2022_Peretzke_Interactive_Fiber_Dissection/data/WhiteCat/WhiteCAT_FA_MD_consti/whiteCAT.xlsx")  # Replace with your actual file name
    # Normalize Label_25 column: convert "TRUE" to 1, anything else to 0
    df['Label_25_normalized'] = df['Label_25'].apply(lambda x: 1 if str(x).strip().upper() == "TRUE" or x == 1 else 0)
    # Create dictionary with record_id as key and normalized Label_25 as value
    record_label_dict = dict(zip(df['record_id'], df['Label_25_normalized']))


    preprocess_dataset(nii_files, unique_ids, record_label_dict, 'median', out_folder='/home/c306h/cluster-data/classification/WhiteCat_median_shape',  num_worker=8,)
    preprocess_dataset(nii_files, unique_ids,record_label_dict, [128,256,256], out_folder='/home/c306h/cluster-data/classification/WhiteCat_1mmiso',  num_worker=8)
    preprocess_dataset(nii_files, unique_ids, record_label_dict, [160,160,160], out_folder='/home/c306h/cluster-data/classification/WhiteCat_fixed160patchsize',  num_worker=8)





