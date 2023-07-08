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
    def __init__(self, imgs_dir, lh_fmri_dir, rh_fmri_dir, transform=None):
        # 保存路径
        self.imgs_dir = imgs_dir
        
        # 图片集
        self.img_list = os.listdir(imgs_dir)
        self.img_list.sort()
        # fmri左右脑数据
        if lh_fmri_dir is not None:
            self.lh_fmri = np.load(lh_fmri_dir)
        if rh_fmri_dir is not None:
            self.rh_fmri = np.load(rh_fmri_dir)
        
        # 如果有转换方法，就保存下来
        self.transform = transform

    def __getitem__(self, index):
        # 图片路径
        img_path = os.path.join(self.imgs_dir, self.img_list[index])
        # 读取图像文件，比如是一个png格式的文件
        image = Image.open(img_path).convert('RGB')
        # 如果有转换方法，就对图像进行转换，比如裁剪、缩放、归一化等
        if self.transform:
            image = self.transform(image)

        if self.lh_fmri is not None and self.lh_fmri is not None:
            lh_fmri = self.lh_fmri[index]
            rh_fmri = self.rh_fmri[index]

            return image, lh_fmri, rh_fmri
        else:
            return image

    def __len__(self):
        return len(self.img_list)