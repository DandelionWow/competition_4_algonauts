import os
from torch.utils.data.dataset import Dataset

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
    def __init__(self, imgs_dir, lh_fmri_dir, rh_fmri_dir):
        super().__init__()
        self.imgs_dir = imgs_dir
        self.lh_fmri_dir = lh_fmri_dir
        self.rh_fmri_dir = rh_fmri_dir

        self.img_list = os.listdir(imgs_dir)


    def __getitem__(self, index):

        return super().__getitem__(index)
    
    def __len__(self):
        count = len(self.images)
        return count