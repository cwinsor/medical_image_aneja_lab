import torch
import numpy as np


class DatasetUWMadison3D(torch.utils.data.Dataset):
    def __init__(self, df, transforms):
        self.df = df
        self.transforms = transforms
        print(df.iloc[0])
        print(df.info())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = np.load(self.df.loc[index].image_path, encoding='bytes')
        mask = np.load(self.df.loc[index].mask_path, encoding='bytes')
        return image, mask
