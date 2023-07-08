'''
alexnet提取img特征，左右脑每个ROI_CLASS分别做线性回归，最后汇总结果

15.0733318042
'''

import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
import pickle

data_dir = '/data/SunYang/datasets/Algonauts_dataset/algonauts_2023_main/algonauts_2023_challenge_data/'
parent_submission_dir = '/data/SunYang/datasets/Algonauts_dataset/algonauts_2023_main/algonauts_2023_challenge_submission'

device = 'cuda:0' 
device = torch.device(device)

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

def get_index(train_img_list, test_img_list):
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

    return idxs_train, idxs_val, idxs_test

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

# 2.2 Extract and downsample image features from AlexNet
# 2.2.1 Load the pretrained AlexNet
model = torch.hub.load('/home/SunYang/.cache/torch/hub/alexnet', 'alexnet', source='local')
model.to(device) # send the model to the chosen device ('cpu' or 'cuda')
model.eval() # set the model to evaluation mode, since you are not training it

train_nodes, _ = get_graph_node_names(model)
print(train_nodes)

model_layer = "features.2" #@param ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"] {allow-input: true}
feature_extractor = create_feature_extractor(model, return_nodes=[model_layer])

def fit_pca(feature_extractor, dataloader):

    # Define PCA parameters
    pca = IncrementalPCA(n_components=100, batch_size=batch_size)

    # Fit PCA to batch
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Fit PCA to batch
        pca.partial_fit(ft.detach().cpu().numpy())
    return pca

def extract_features(feature_extractor, dataloader, pca):

    features = []
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Apply PCA transform
        ft = pca.transform(ft.cpu().detach().numpy())
        features.append(ft)
    return np.vstack(features)


if __name__ == '__main__':

    for subj in range(1, 5):
        print('subj: '+format(subj, '02')+'\n')

        # 参数
        args = argObj(data_dir, parent_submission_dir, subj)
        # Stimulus images
        train_img_dir  = os.path.join(args.data_dir, 'training_split', 'training_images')
        test_img_dir  = os.path.join(args.data_dir, 'test_split', 'test_images')
        train_img_list = os.listdir(train_img_dir)
        train_img_list.sort()
        test_img_list = os.listdir(test_img_dir)
        test_img_list.sort()
        print('Training images: ' + str(len(train_img_list)))
        print('Test images: ' + str(len(test_img_list)))
        print('\n')
        # index
        idxs_train, idxs_val, idxs_test = get_index(train_img_list, test_img_list)
        
        pkl_path = '/data/SunYang/datasets/Algonauts_dataset/pkl_alexnet'
        train_pkl_path = os.path.join(pkl_path, 'subj'+format(subj, '02'), "features_train.pkl")
        val_pkl_path = os.path.join(pkl_path, 'subj'+format(subj, '02'), "features_val.pkl")
        test_pkl_path = os.path.join(pkl_path, 'subj'+format(subj, '02'), "features_test.pkl")
        if os.path.exists(train_pkl_path) and os.path.exists(val_pkl_path) and os.path.exists(test_pkl_path):
            with open(train_pkl_path, 'rb') as f:
                features_train = pickle.load(f)
            with open(val_pkl_path, 'rb') as f:
                features_val = pickle.load(f)
            with open(test_pkl_path, 'rb') as f:
                features_test = pickle.load(f)
        else:
            # 加载数据集
            batch_size = 256
            # Get the paths of all image files
            train_imgs_paths = sorted(list(Path(train_img_dir).iterdir()))
            test_imgs_paths = sorted(list(Path(test_img_dir).iterdir()))
            # The DataLoaders contain the ImageDataset class
            train_imgs_dataloader = DataLoader(
                ImageDataset(train_imgs_paths, idxs_train, transform), 
                batch_size=batch_size,
            )
            val_imgs_dataloader = DataLoader(
                ImageDataset(train_imgs_paths, idxs_val, transform), 
                batch_size=batch_size,
            )
            test_imgs_dataloader = DataLoader(
                ImageDataset(test_imgs_paths, idxs_test, transform), 
                batch_size=batch_size
            )
            
            # 拟合pca
            pca = fit_pca(feature_extractor, train_imgs_dataloader)
            
            # 特征提取
            features_train = extract_features(feature_extractor, train_imgs_dataloader, pca)
            features_val = extract_features(feature_extractor, val_imgs_dataloader, pca)
            features_test = extract_features(feature_extractor, test_imgs_dataloader, pca)
            
            # 保存特征
            subj_path = "subj" + format(subj, '02')
            if os.path.exists(os.path.join(pkl_path, subj_path)) is False:
                os.mkdir(os.path.join(pkl_path, subj_path))
            with open(os.path.join(pkl_path, subj_path, "features_train.pkl"), 'wb') as f:
                pickle.dump(features_train, f)
            with open(os.path.join(pkl_path, subj_path, "features_val.pkl"), 'wb') as f:
                pickle.dump(features_val, f)
            with open(os.path.join(pkl_path, subj_path, "features_test.pkl"), 'wb') as f:
                pickle.dump(features_test, f)

        # roi映射
        # roi_map_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'mapping_prf-visualrois.npy')
        # roi_map = np.load(roi_map_dir, allow_pickle=True).item()

        fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
        lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
        rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))
        lh_fmri_train = lh_fmri[idxs_train]
        lh_fmri_val = lh_fmri[idxs_val]
        rh_fmri_train = rh_fmri[idxs_train]
        rh_fmri_val = rh_fmri[idxs_val]

        # lh roi prf_visualrois
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'lh.prf-visualrois_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        lh_prf_visualrois = np.multiply(lh_fmri_train, challenge_roi)
        # rh roi prf_visualrois
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'rh.prf-visualrois_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        rh_prf_visualrois = np.multiply(rh_fmri_train, challenge_roi)
        # Fit linear regressions on the training data
        reg_lh_prf_visualrois = LinearRegression().fit(features_train, lh_prf_visualrois)
        reg_rh_prf_visualrois = LinearRegression().fit(features_train, rh_prf_visualrois)
        # Use fitted linear regressions to predict the validation and test fMRI data
        lh_fmri_prf_visualrois_test_pred = reg_lh_prf_visualrois.predict(features_test)
        rh_fmri_prf_visualrois_test_pred = reg_rh_prf_visualrois.predict(features_test)
        
        # lh roi floc_bodies
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'lh.floc-bodies_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        lh_floc_bodies = np.multiply(lh_fmri_train, challenge_roi)
        # rh roi floc_bodies
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'rh.floc-bodies_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        rh_floc_bodies = np.multiply(rh_fmri_train, challenge_roi)
        # Fit linear regressions on the training data
        reg_lh_floc_bodies = LinearRegression().fit(features_train, lh_floc_bodies)
        reg_rh_floc_bodies = LinearRegression().fit(features_train, rh_floc_bodies)
        # Use fitted linear regressions to predict the validation and test fMRI data
        lh_fmri_floc_bodies_test_pred = reg_lh_floc_bodies.predict(features_test)
        rh_fmri_floc_bodies_test_pred = reg_rh_floc_bodies.predict(features_test)

        # lh roi floc_faces
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'lh.floc-faces_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        lh_floc_faces = np.multiply(lh_fmri_train, challenge_roi)
        # rh roi floc_faces
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'rh.floc-faces_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        rh_floc_faces = np.multiply(rh_fmri_train, challenge_roi)
        # Fit linear regressions on the training data
        reg_lh_floc_faces = LinearRegression().fit(features_train, lh_floc_faces)
        reg_rh_floc_faces = LinearRegression().fit(features_train, rh_floc_faces)
        # Use fitted linear regressions to predict the validation and test fMRI data
        lh_fmri_floc_faces_test_pred = reg_lh_floc_faces.predict(features_test)
        rh_fmri_floc_faces_test_pred = reg_rh_floc_faces.predict(features_test)

        # lh roi floc_places
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'lh.floc-places_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        lh_floc_places = np.multiply(lh_fmri_train, challenge_roi)
        # rh roi floc_places
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'rh.floc-places_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        rh_floc_places = np.multiply(rh_fmri_train, challenge_roi)
        # Fit linear regressions on the training data
        reg_lh_floc_places = LinearRegression().fit(features_train, lh_floc_places)
        reg_rh_floc_places = LinearRegression().fit(features_train, rh_floc_places)
        # Use fitted linear regressions to predict the validation and test fMRI data
        lh_fmri_floc_places_test_pred = reg_lh_floc_places.predict(features_test)
        rh_fmri_floc_places_test_pred = reg_rh_floc_places.predict(features_test)

        # lh roi floc_words
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'lh.floc-words_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        lh_floc_words = np.multiply(lh_fmri_train, challenge_roi)
        # rh roi floc_words
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'rh.floc-words_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        rh_floc_words = np.multiply(rh_fmri_train, challenge_roi)
        # Fit linear regressions on the training data
        reg_lh_floc_words = LinearRegression().fit(features_train, lh_floc_words)
        reg_rh_floc_words = LinearRegression().fit(features_train, rh_floc_words)
        # Use fitted linear regressions to predict the validation and test fMRI data
        lh_fmri_floc_words_test_pred = reg_lh_floc_words.predict(features_test)
        rh_fmri_floc_words_test_pred = reg_rh_floc_words.predict(features_test)

        # lh roi streams
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'lh.streams_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        lh_streams = np.multiply(lh_fmri_train, challenge_roi)
        # rh roi streams
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'rh.streams_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        rh_streams = np.multiply(rh_fmri_train, challenge_roi)
        # Fit linear regressions on the training data
        reg_lh_streams = LinearRegression().fit(features_train, lh_streams)
        reg_rh_streams = LinearRegression().fit(features_train, rh_streams)
        # Use fitted linear regressions to predict the validation and test fMRI data
        lh_fmri_streams_test_pred = reg_lh_streams.predict(features_test)
        rh_fmri_streams_test_pred = reg_rh_streams.predict(features_test)

        # 汇总test预测结果 lh
        lh_fmri_test_pred = np.add(lh_fmri_prf_visualrois_test_pred, lh_fmri_floc_bodies_test_pred)
        lh_fmri_test_pred = np.add(lh_fmri_test_pred, lh_fmri_floc_faces_test_pred)
        lh_fmri_test_pred = np.add(lh_fmri_test_pred, lh_fmri_floc_places_test_pred)
        lh_fmri_test_pred = np.add(lh_fmri_test_pred, lh_fmri_floc_words_test_pred)
        lh_fmri_test_pred = np.add(lh_fmri_test_pred, lh_fmri_streams_test_pred)
        # 汇总test预测结果 rh
        rh_fmri_test_pred = np.add(rh_fmri_prf_visualrois_test_pred, rh_fmri_floc_bodies_test_pred)
        rh_fmri_test_pred = np.add(rh_fmri_test_pred, rh_fmri_floc_faces_test_pred)
        rh_fmri_test_pred = np.add(rh_fmri_test_pred, rh_fmri_floc_places_test_pred)
        rh_fmri_test_pred = np.add(rh_fmri_test_pred, rh_fmri_floc_words_test_pred)
        rh_fmri_test_pred = np.add(rh_fmri_test_pred, rh_fmri_streams_test_pred)

        # 保存测试结果，用于上传
        lh_fmri_test_pred = lh_fmri_test_pred.astype(np.float32)
        rh_fmri_test_pred = rh_fmri_test_pred.astype(np.float32)
        subject_submission_dir = os.path.join(parent_submission_dir,'subj'+format(subj, '02'))
        if os.path.exists(subject_submission_dir) is False:
            os.mkdir(subject_submission_dir)
        np.save(os.path.join(subject_submission_dir, 'lh_pred_test.npy'), lh_fmri_test_pred)
        np.save(os.path.join(subject_submission_dir, 'rh_pred_test.npy'), rh_fmri_test_pred)

