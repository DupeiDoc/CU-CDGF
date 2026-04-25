import torch.utils.data as data
import numpy as np
from PIL import Image

class CarotidDataset2D(data.Dataset):
    """2D carotid ultrasound dataset with image and mask paths."""
    def __init__(self, dataframe, img_transform=None, mask_transform=None):
        self.dataframe = dataframe
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['img']
        label = row.get('label', 0)  # 如果有标签列
        # 掩膜路径约定：原图名去掉 .png 后加 -ALLMASK.png
        mask_path = img_path.split('.png')[0] + '-ALLMASK.png'

        data_dict = {
            "image": img_path,
            "mask": mask_path,
            "label": label
        }

        if self.img_transform:
            data_dict = self.img_transform(data_dict)

        return data_dict, img_path   # 返回(img_path)便于调试
