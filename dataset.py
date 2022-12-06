import torch, cv2
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import *
from albumentations.pytorch.transforms import *

from preprocessing import *

class SiimDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir='./jpg_form', transform=None):
        super().__init__()
        self.df = df
        self.length = len(df)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return self.length 
        
    def __getitem__(self, index):
        # default img size used: 512x512
        data = self.df.iloc[index]
        img_path = os.path.join(self.root_dir, 'train', data.id + '.jpg')
        image = cv2.imread(img_path)
        landmarks = data.label.strip('[]').split(', ')
        landmarks = np.array(landmarks).astype(np.float)
        landmarks = torch.from_numpy(landmarks)
        
        if self.transform:
            image = self.transform(image=image)['image']
        return image, landmarks
        

