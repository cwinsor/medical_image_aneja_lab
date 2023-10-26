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
        image_raw = np.load(self.df.loc[index].image_path, encoding='bytes')
        mask_raw = np.load(self.df.loc[index].mask_path, encoding='bytes')
        id = self.df.loc[index, 'id']
        case = self.df.loc[index, 'case']
        day = self.df.loc[index, 'day']

        id_case_day = list(self.df.loc[index, ['id','case','day']])
        # ZONA - we spend a lot of code to reshape the 3D voxel into a fixed-size target
        # using crop and padding...
        # BUT what we really need to do is use the scale factors provided in the dataset (image name)
        # which identifies millimeter dimensions,
        # and the https://torchio.readthedocs.io/transforms/preprocessing.html#torchio.transforms.Resample
        # which also used millimeter dimensions...

        if image_raw.shape != mask_raw.shape:
            assert False, "whoh - image and mask shapes not the same !!!"

        # return (image_raw, mask_raw, id, case, day)
        return image_raw, mask_raw, case, day