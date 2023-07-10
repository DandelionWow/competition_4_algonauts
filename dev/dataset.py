# 导入pytorch相关的库
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class PreProcessingDataset(Dataset):
    def __init__(self, dataset_path, imgs_dir):
        super().__init__()
        self.path = dataset_path
        self.img_list = os.listdir(imgs_dir)


    def __getitem__(self, index):

        return super().__getitem__(index)
    
    def __len__(self):
        count = len(self.images)
        return count

class AlgonautsDataset(Dataset):
    def __init__(self, imgs_dir, lh_fmri_dir=None, rh_fmri_dir=None, transform=None):
        # 图片特征
        self.imgs_feature = np.load(imgs_dir)
        # fmri左右脑数据
        self.lh_fmri = np.load(lh_fmri_dir) if lh_fmri_dir is not None else None
        self.rh_fmri = np.load(rh_fmri_dir) if rh_fmri_dir is not None else None
        
        # 如果有转换方法，就保存下来
        self.transform = transform

    def __getitem__(self, index):
        # 读取图像特征
        img_feature = self.imgs_feature[index]
        # 如果有转换方法，就对图像进行转换，比如裁剪、缩放、归一化等
        if self.transform:
            img_feature = self.transform(img_feature)

        if self.lh_fmri is not None and self.lh_fmri is not None:
            lh_fmri = self.lh_fmri[index]
            rh_fmri = self.rh_fmri[index]

            return img_feature, lh_fmri, rh_fmri
        else:
            return img_feature

    def __len__(self):
        return len(self.imgs_feature)