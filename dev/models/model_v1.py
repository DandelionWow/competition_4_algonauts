'''
clip提取图像特征，模型使用线性回归
'''

import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
# from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
import clip
import pickle

data_dir = '/data/SunYang/datasets/Algonauts_dataset/algonauts_2023_main/algonauts_2023_challenge_data/'
parent_submission_dir = '/data/SunYang/datasets/Algonauts_dataset/algonauts_2023_main/algonauts_2023_challenge_submission'

device = 'cuda:1' 
device = torch.device(device)

subj = 1

class argObj:
  def __init__(self, data_dir, parent_submission_dir, subj):
    
    self.subj = format(subj, '02')
    self.data_dir = os.path.join(data_dir, 'subj'+self.subj)
    self.parent_submission_dir = parent_submission_dir
    self.subject_submission_dir = os.path.join(self.parent_submission_dir,
        'subj'+self.subj)

    # Create the submission directory if not existing
    if not os.path.isdir(self.subject_submission_dir):
        os.makedirs(self.subject_submission_dir)

args = argObj(data_dir, parent_submission_dir, subj)

# Stimulus images
train_img_dir  = os.path.join(args.data_dir, 'training_split', 'training_images')
test_img_dir  = os.path.join(args.data_dir, 'test_split', 'test_images')
# Create lists will all training and test image file names, sorted
train_img_list = os.listdir(train_img_dir)
train_img_list.sort()
test_img_list = os.listdir(test_img_dir)
test_img_list.sort()
print('\n')
print('Training images: ' + str(len(train_img_list)))
print('Test images: ' + str(len(test_img_list)))

# 2.1.1 Create the training, validation and test partitions indices
rand_seed = 5
np.random.seed(rand_seed)

# Calculate how many stimulus images correspond to 90% of the training data
num_train = int(np.round(len(train_img_list) / 100 * 90))
# Shuffle all training stimulus images
idxs = np.arange(len(train_img_list))
np.random.shuffle(idxs)
# Assign 90% of the shuffled stimulus images to the training partition,
# and 10% to the test partition
idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]
# No need to shuffle or split the test stimulus images
idxs_test = np.arange(len(test_img_list))

print('\n')
print('Training stimulus images: ' + format(len(idxs_train)))
print('Validation stimulus images: ' + format(len(idxs_val)))
print('Test stimulus images: ' + format(len(idxs_test)))

# 2.1.2 Create the training, validation and test image partitions DataLoaders
transform = transforms.Compose([
    transforms.Resize((224,224)), # resize the images to 224x24 pixels
    transforms.ToTensor(), # convert the images to a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
])
class ImageDataset(Dataset):
    def __init__(self, imgs_paths, idxs, transform):
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.transform = transform

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        if self.transform:
            img = self.transform(img).to(device)
        return img
batch_size = 300
# Get the paths of all image files
train_imgs_paths = sorted(list(Path(train_img_dir).iterdir()))
test_imgs_paths = sorted(list(Path(test_img_dir).iterdir()))
# The DataLoaders contain the ImageDataset class
train_imgs_dataloader = DataLoader(
    ImageDataset(train_imgs_paths, idxs_train, transform), 
    batch_size=batch_size
)
val_imgs_dataloader = DataLoader(
    ImageDataset(train_imgs_paths, idxs_val, transform), 
    batch_size=batch_size
)
test_imgs_dataloader = DataLoader(
    ImageDataset(test_imgs_paths, idxs_test, transform), 
    batch_size=batch_size
)
print('\n')

# # 2.2 Extract and downsample image features from AlexNet
# # 2.2.1 Load the pretrained AlexNet
# model = torch.hub.load('/home/SunYang/.cache/torch/hub/alexnet', 'alexnet', source='local')
# model.to(device) # send the model to the chosen device ('cpu' or 'cuda')
# model.eval() # set the model to evaluation mode, since you are not training it

# train_nodes, _ = get_graph_node_names(model)
# print(train_nodes)

# model_layer = "features.2" #@param ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"] {allow-input: true}
# feature_extractor = create_feature_extractor(model, return_nodes=[model_layer])

# def fit_pca(feature_extractor, dataloader):

#     # Define PCA parameters
#     pca = IncrementalPCA(n_components=100, batch_size=batch_size)

#     # Fit PCA to batch
#     for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
#         # Extract features
#         ft = feature_extractor(d)
#         # Flatten the features
#         ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
#         # Fit PCA to batch
#         pca.partial_fit(ft.detach().cpu().numpy())
#     return pca
# pca = fit_pca(feature_extractor, train_imgs_dataloader)

# def extract_features(feature_extractor, dataloader, pca):

#     features = []
#     for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
#         # Extract features
#         ft = feature_extractor(d)
#         # Flatten the features
#         ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
#         # Apply PCA transform
#         ft = pca.transform(ft.cpu().detach().numpy())
#         features.append(ft)
#     return np.vstack(features)

# features_train = extract_features(feature_extractor, train_imgs_dataloader, pca)
# features_val = extract_features(feature_extractor, val_imgs_dataloader, pca)
# features_test = extract_features(feature_extractor, test_imgs_dataloader, pca)

# print('\nTraining images features:')
# print(features_train.shape)
# print('(Training stimulus images × PCA features)')

# print('\nValidation images features:')
# print(features_val.shape)
# print('(Validation stimulus images × PCA features)')

# print('\nTest images features:')
# print(features_val.shape)
# print('(Test stimulus images × PCA features)')

def preprocess_img(subj, dir):
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    data_dir = '/data/SunYang/datasets/Algonauts_dataset/algonauts_2023_main/algonauts_2023_challenge_data'
    subj = 'subj0{0}'.format(subj)
    img_dir = os.path.join(data_dir, subj, dir)
    img_list = os.listdir(img_dir)
    img_list.sort()

    N_IMG = len(img_list)
    print("#images:", N_IMG)
    all_img_features = np.zeros((N_IMG, 512))
    with torch.no_grad():
        for i in tqdm(range(N_IMG)):
            im = preprocess(Image.open(os.path.join(img_dir, img_list[i]))).unsqueeze(0).to(device)
            all_img_features[i] = model.encode_image(im).cpu().numpy()
    return N_IMG, all_img_features

def preprocess_fMRI(subj, dir):
    subj = 'subj0{0}'.format(subj)
    data_dir = '/data/SunYang/datasets/Algonauts_dataset/algonauts_2023_main/algonauts_2023_challenge_data'
    fmir_dir = os.path.join(data_dir, subj, dir)
    lh_fmir = np.load(os.path.join(fmir_dir, 'lh_training_fmri.npy'))
    rh_fmir = np.load(os.path.join(fmir_dir, 'rh_training_fmri.npy'))
    return lh_fmir, rh_fmir

def consolidate_subj(subj):
    pkl_path = "/data/SunYang/datasets/Algonauts_dataset/pkl"
    train_img_dir  = os.path.join('training_split', 'training_images')
    test_img_dir  = os.path.join('test_split', 'test_images')
    train_fmir_dir = os.path.join('training_split','training_fmri')
    
    ### get train_data and valid data
    ### 划分比例: 90% 10%
    train_data = {}
    valid_data = {}
    
    N_IMG, subj_img_features = preprocess_img(subj, train_img_dir)
    lh_fmri, rh_fmri = preprocess_fMRI(subj, train_fmir_dir)
    num_train = int(np.round(N_IMG / 100 * 90))
    idxs = np.arange(N_IMG)
    np.random.shuffle(idxs)
    idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]

    train_data['img'] = subj_img_features[idxs_train]
    train_data['lh_fmri'] = lh_fmri[idxs_train]
    train_data['rh_fmri'] = rh_fmri[idxs_train]

    valid_data['img'] = subj_img_features[idxs_val]
    valid_data['lh_fmri'] = lh_fmri[idxs_val]
    valid_data['rh_fmri'] = rh_fmri[idxs_val]
    
    test_data = {}
    # pca = fit_pca(subj, test_img_dir)
    _, subj_img_features_test = preprocess_img(subj, test_img_dir)
    test_data['img'] = subj_img_features_test

    # sava tarin_data valid_data and test_data
    subj_path = "subj" + format(subj, '02')
    if os.path.exists(os.path.join(pkl_path, subj_path)) is False:
        os.mkdir(os.path.join(pkl_path, subj_path))
    with open(os.path.join(pkl_path, subj_path, "train.pkl"), 'wb') as f:
        pickle.dump(train_data, f)

    with open(os.path.join(pkl_path, subj_path, "val.pkl"), 'wb') as f:
        pickle.dump(valid_data, f)

    with open(os.path.join(pkl_path, subj_path, "test.pkl"), 'wb') as f:
        pickle.dump(test_data, f)

    return train_data, valid_data, test_data

if __name__ == '__main__':

    for subj in range(1, 9):
        train_pkl_path = os.path.join('/data/SunYang/datasets/Algonauts_dataset/pkl', 'subj'+format(subj, '02'), "train.pkl")
        val_pkl_path = os.path.join('/data/SunYang/datasets/Algonauts_dataset/pkl', 'subj'+format(subj, '02'), "val.pkl")
        test_pkl_path = os.path.join('/data/SunYang/datasets/Algonauts_dataset/pkl', 'subj'+format(subj, '02'), "test.pkl")
        if os.path.exists(train_pkl_path) and os.path.exists(val_pkl_path) and os.path.exists(test_pkl_path):
            with open(train_pkl_path, 'rb') as f:
                train_data = pickle.load(f)
            with open(val_pkl_path, 'rb') as f:
                valid_data = pickle.load(f)
            with open(test_pkl_path, 'rb') as f:
                test_data = pickle.load(f)
        else:
            train_data, valid_data, test_data = consolidate_subj(subj)

        # Fit linear regressions on the training data
        reg_lh = LinearRegression().fit(train_data['img'], train_data['lh_fmri'])
        reg_rh = LinearRegression().fit(train_data['img'], train_data['rh_fmri'])
        # Use fitted linear regressions to predict the validation and test fMRI data
        lh_fmri_val_pred = reg_lh.predict(valid_data['img'])
        lh_fmri_test_pred = reg_lh.predict(test_data['img'])
        rh_fmri_val_pred = reg_rh.predict(valid_data['img'])
        rh_fmri_test_pred = reg_rh.predict(test_data['img'])

        # 保存测试结果，用于上传
        lh_fmri_test_pred = lh_fmri_test_pred.astype(np.float32)
        rh_fmri_test_pred = rh_fmri_test_pred.astype(np.float32)
        subject_submission_dir = os.path.join(parent_submission_dir,'subj'+format(subj, '02'))
        if os.path.exists(subject_submission_dir) is False:
            os.mkdir(subject_submission_dir)
        np.save(os.path.join(subject_submission_dir, 'lh_pred_test.npy'), lh_fmri_test_pred)
        np.save(os.path.join(subject_submission_dir, 'rh_pred_test.npy'), rh_fmri_test_pred)

