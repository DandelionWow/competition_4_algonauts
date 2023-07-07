'''
clip提取图像特征，左右脑每个ROI分别做线性回归，最后汇总结果

37.5407612557	
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

device = 'cuda:2' 
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

def preprocess_img(subj, dir):
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
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

    for subj in range(6, 9):
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

        # roi映射
        # roi_map_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'mapping_prf-visualrois.npy')
        # roi_map = np.load(roi_map_dir, allow_pickle=True).item()
        
        # lh roi prf_visualrois
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'lh.prf-visualrois_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        lh_prf_visualrois = np.multiply(train_data['lh_fmri'], challenge_roi)
        # rh roi prf_visualrois
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'rh.prf-visualrois_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        rh_prf_visualrois = np.multiply(train_data['rh_fmri'], challenge_roi)
        # Fit linear regressions on the training data
        reg_lh_prf_visualrois = LinearRegression().fit(train_data['img'], lh_prf_visualrois)
        reg_rh_prf_visualrois = LinearRegression().fit(train_data['img'], rh_prf_visualrois)
        # Use fitted linear regressions to predict the validation and test fMRI data
        lh_fmri_prf_visualrois_test_pred = reg_lh_prf_visualrois.predict(test_data['img'])
        rh_fmri_prf_visualrois_test_pred = reg_rh_prf_visualrois.predict(test_data['img'])
        
        # lh roi floc_bodies
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'lh.floc-bodies_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        lh_floc_bodies = np.multiply(train_data['lh_fmri'], challenge_roi)
        # rh roi floc_bodies
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'rh.floc-bodies_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        rh_floc_bodies = np.multiply(train_data['rh_fmri'], challenge_roi)
        # Fit linear regressions on the training data
        reg_lh_floc_bodies = LinearRegression().fit(train_data['img'], lh_floc_bodies)
        reg_rh_floc_bodies = LinearRegression().fit(train_data['img'], rh_floc_bodies)
        # Use fitted linear regressions to predict the validation and test fMRI data
        lh_fmri_floc_bodies_test_pred = reg_lh_floc_bodies.predict(test_data['img'])
        rh_fmri_floc_bodies_test_pred = reg_rh_floc_bodies.predict(test_data['img'])

        # lh roi floc_faces
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'lh.floc-faces_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        lh_floc_faces = np.multiply(train_data['lh_fmri'], challenge_roi)
        # rh roi floc_faces
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'rh.floc-faces_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        rh_floc_faces = np.multiply(train_data['rh_fmri'], challenge_roi)
        # Fit linear regressions on the training data
        reg_lh_floc_faces = LinearRegression().fit(train_data['img'], lh_floc_faces)
        reg_rh_floc_faces = LinearRegression().fit(train_data['img'], rh_floc_faces)
        # Use fitted linear regressions to predict the validation and test fMRI data
        lh_fmri_floc_faces_test_pred = reg_lh_floc_faces.predict(test_data['img'])
        rh_fmri_floc_faces_test_pred = reg_rh_floc_faces.predict(test_data['img'])

        # lh roi floc_places
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'lh.floc-places_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        lh_floc_places = np.multiply(train_data['lh_fmri'], challenge_roi)
        # rh roi floc_places
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'rh.floc-places_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        rh_floc_places = np.multiply(train_data['rh_fmri'], challenge_roi)
        # Fit linear regressions on the training data
        reg_lh_floc_places = LinearRegression().fit(train_data['img'], lh_floc_places)
        reg_rh_floc_places = LinearRegression().fit(train_data['img'], rh_floc_places)
        # Use fitted linear regressions to predict the validation and test fMRI data
        lh_fmri_floc_places_test_pred = reg_lh_floc_places.predict(test_data['img'])
        rh_fmri_floc_places_test_pred = reg_rh_floc_places.predict(test_data['img'])

        # lh roi floc_words
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'lh.floc-words_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        lh_floc_words = np.multiply(train_data['lh_fmri'], challenge_roi)
        # rh roi floc_words
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'rh.floc-words_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        rh_floc_words = np.multiply(train_data['rh_fmri'], challenge_roi)
        # Fit linear regressions on the training data
        reg_lh_floc_words = LinearRegression().fit(train_data['img'], lh_floc_words)
        reg_rh_floc_words = LinearRegression().fit(train_data['img'], rh_floc_words)
        # Use fitted linear regressions to predict the validation and test fMRI data
        lh_fmri_floc_words_test_pred = reg_lh_floc_words.predict(test_data['img'])
        rh_fmri_floc_words_test_pred = reg_rh_floc_words.predict(test_data['img'])

        # lh roi streams
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'lh.streams_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        lh_streams = np.multiply(train_data['lh_fmri'], challenge_roi)
        # rh roi streams
        challenge_roi_class_dir = os.path.join(data_dir, 'subj'+format(subj, '02'), 'roi_masks', 'rh.streams_challenge_space.npy')
        challenge_roi_class = np.load(challenge_roi_class_dir)
        challenge_roi = np.asarray(challenge_roi_class != 0, dtype=int)
        rh_streams = np.multiply(train_data['rh_fmri'], challenge_roi)
        # Fit linear regressions on the training data
        reg_lh_streams = LinearRegression().fit(train_data['img'], lh_streams)
        reg_rh_streams = LinearRegression().fit(train_data['img'], rh_streams)
        # Use fitted linear regressions to predict the validation and test fMRI data
        lh_fmri_streams_test_pred = reg_lh_streams.predict(test_data['img'])
        rh_fmri_streams_test_pred = reg_rh_streams.predict(test_data['img'])

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

