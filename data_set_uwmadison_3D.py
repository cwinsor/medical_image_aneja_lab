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

        sh, sw, sd = image_raw.shape
        th, tw, td = 40, 40, 40

        def pad_and_crop_idx(source_size, target_size):
            if source_size < target_size:  # pick the smaller
                the_size = source_size
            else:
                the_size = target_size

            half_size = the_size // 2
            source_mid_idx = source_size // 2
            target_mid_idx = target_size // 2

            source_start_idx = source_mid_idx - half_size
            source_end_idx = source_start_idx + the_size - 1
            target_start_idx = target_mid_idx - half_size
            target_end_idx = target_start_idx + the_size - 1

            return source_start_idx, source_end_idx, target_start_idx, target_end_idx

        src_start_h, src_end_h, tar_start_h, tar_end_h = pad_and_crop_idx(sh, th)
        src_start_w, src_end_w, tar_start_w, tar_end_w = pad_and_crop_idx(sw, tw)
        src_start_d, src_end_d, tar_start_d, tar_end_d = pad_and_crop_idx(sd, td)

        image = np.zeros((th, tw, td))
        image[tar_start_h:tar_end_h+1, tar_start_w:tar_end_w+1, tar_start_d:tar_end_d+1] = (
            image_raw[src_start_h:src_end_h+1, src_start_w:src_end_w+1, src_start_d:src_end_d+1])

        mask = np.zeros((th, tw, td))
        mask[tar_start_h:tar_end_h+1, tar_start_w:tar_end_w+1, tar_start_d:tar_end_d+1] = (
            mask_raw[src_start_h:src_end_h+1, src_start_w:src_end_w+1, src_start_d:src_end_d+1])

        return (image, mask, id, case, day)
