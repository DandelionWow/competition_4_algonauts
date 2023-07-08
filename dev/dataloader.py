# dataloader.py: create a data loader object for the custom dataset
import os
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from dataset import AlgonautsDataset

# 训练集
def create_data_loader(config):
    # create a transform function to resize and normalize the images
    transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    imgs_dir = os.path.join(config['dataset_path'], config['imgs_dir'])
    lh_fmri_dir = os.path.join(config['dataset_path'], config['lh_fmri_dir'])
    rh_fmri_dir = os.path.join(config['dataset_path'], config['rh_fmri_dir'])

    # create a custom dataset object with the given image directory and annotation file
    dataset = AlgonautsDataset(imgs_dir=imgs_dir, lh_fmri_dir=lh_fmri_dir, 
                               rh_fmri_dir=rh_fmri_dir, transform=transform)
    
    # get the size of the whole dataset
    dataset_size = len(dataset)
    # calculate the lengths of the train and validation subsets
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    # split the dataset into train and validation subsets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # create a data loader object for the train subset with the parameters from config file
    train_data_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    # create a data loader object for the validation subset with the parameters from config file
    val_data_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    return train_data_loader, val_data_loader

# 测试集
def create_test_data_loader(config):
    # create a transform function to resize and normalize the images
    transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    imgs_dir = os.path.join(config['dataset_path'], config['test_imgs_dir'])

    # create a custom dataset object with the given image directory and annotation file
    dataset = AlgonautsDataset(imgs_dir=imgs_dir, lh_fmri_dir=None, 
                               rh_fmri_dir=None, transform=transform)
    
    # create a data loader object for the train subset with the parameters from config file
    test_data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    return test_data_loader