import torch
import numpy as np


class DatasetUWMadison2D(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None):
        self.df = df
        self.label = label
        self.img_paths = df['image_path'].tolist()
        self.msk_paths = df['mask_path'].tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        index = 60  # zona
        img_3d = []
        msk_3d = []
        for slice in range(2):
            img, msk = self.get_single_image_mask(index)
            img_3d = img_3d.append(img)
            msk_3d = msk_3d.append(msk)
        print("here")


    def get_single_image_mask(self, index):
        img_path = self.img_paths[index]
        img = []
        img = self.load_img(img_path)

        if self.label:
            msk_path = self.msk_paths[index]
            msk = self.load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img = data['image']
                msk = data['mask']
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img = data['image']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img)

    def load_img(self, path):
        img = np.load(path)
        img = img.astype('float32')  # original is uint16
        mx = np.max(img)
        if mx:
            img /= mx  # scale image to [0, 1]
        return img

    def load_msk(self, path):
        msk = np.load(path)
        msk = msk.astype('float32')
        msk /= 255.0
        return msk
