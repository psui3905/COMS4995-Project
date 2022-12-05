
import os, pydicom
import pandas as pd 
import numpy as np

from PIL import Image
from tqdm.auto import tqdm
from pydicom.pixel_data_handlers.util import apply_voi_lut


def read_xray(path, voi_lut = True, fix_monochrome = True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to 
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)
    
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    return im

def convert_to_jpg(study_level_csv, meta_csv_p='./meta.csv', 
                    save_dir='./jpg_form', dataset_p='../siim-covid19-detection',
                    img_size=512):
    train_csv = pd.read_csv(study_level_csv)
    image_id = []
    dim0 = []
    dim1 = []
    splits = []
    labels = []
    

    for split in ['test', 'train']:
        save_dir = save_dir + f'/{split}/'
        os.makedirs(save_dir, exist_ok=True)
    
    data_path = dataset_p + f'/{split}'
    for root, dirs, filenames in tqdm(os.walk(data_path)):
        for file in filenames:
            # set keep_ratio=True to have original aspect ratio
            csv_id = file.split('/')[0] + '_study'
            id_row = train_csv.loc[train_csv['id'] == csv_id]
            id_row = id_row.loc[:, id_row.columns != 'id']
            label = id_row.values.tolist()
            
            xray = read_xray(os.path.join(root, file))
            im = resize(xray, size=img_size)  
            im.save(os.path.join(save_dir, file.replace('dcm', 'jpg')))

            image_id.append(file.replace('.dcm', ''))
            dim0.append(xray.shape[0])
            dim1.append(xray.shape[1])
            labels.append(label)
            splits.append(split)
            
    df = pd.DataFrame.from_dict({'id': image_id, 'dim0': dim0, 'dim1': dim1, 'label': labels, 'split': splits})
    df.to_csv(meta_csv_p, index=False)
    return meta_csv_p



