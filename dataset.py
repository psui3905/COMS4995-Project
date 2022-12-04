import torch, cv2
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *

from preprocessing import *

class SiimDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
        self.length = len(df)
    
    def __len__(self):
        return self.length 
        
    def __getitem__(self, index):
        # default img size used: 512x512
        data = self.df.iloc[index]
        image = cv2.imread('train/' + data.id + '.jpg')
        label = np.array(data.label)
        return image, label
        
if __name__ == '__main__':
    meta_csv_path = './meta.csv'
    train_csv_path = '../siim-covid19-detection/train_image_level.csv'
    convert_to_jpg(train_csv_path)
    png_df = pd.read_csv(meta_csv_path)
    dataset = SiimDataset(png_df)
    img, label = dataset[0]
    print(img)
    # cv2.imshow('sample', img)
    print(label)