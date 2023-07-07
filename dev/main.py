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

hemisphere_left = 'left'
hemisphere_right = 'right'

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

def get_roi_mapping(data_dir, subj, roi_class, roi):
    # roi映射
    roi_map_dir = os.path.join(data_dir, subj, 'roi_masks', 'mapping_'+roi_class+'.npy')
    roi_map = np.load(roi_map_dir, allow_pickle=True).item()
    roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]
    return roi_mapping

def get_fmri_of_roi(data_dir, subj, hemisphere, train_data, roi_class, roi_mapping):
    # 在challenge_space中划分出roi的点
    challenge_roi_class_dir = os.path.join(data_dir, subj, 'roi_masks', 
                                           hemisphere[0]+'h.'+roi_class+'_challenge_space.npy')
    challenge_roi_class = np.load(challenge_roi_class_dir)
    challenge_roi = np.asarray(challenge_roi_class == roi_mapping, dtype=int)

    # 仅保留roi的fmri数据（非roi归0）
    fmri = np.multiply(train_data[hemisphere[0]+'h_fmri'], challenge_roi)

    return fmri

def linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, roi_mapping):
    lh_fmri_roi = get_fmri_of_roi(data_dir, subj, hemisphere_left, train_data, roi_class, roi_mapping)
    rh_fmri_roi = get_fmri_of_roi(data_dir, subj, hemisphere_right, train_data, roi_class, roi_mapping)
    
    # Fit linear regressions on the training data
    reg_lh = LinearRegression().fit(train_data['img'], lh_fmri_roi)
    reg_rh = LinearRegression().fit(train_data['img'], rh_fmri_roi)
    # Use fitted linear regressions to predict the validation and test fMRI data
    lh_test_pred = reg_lh.predict(test_data['img'])
    rh_test_pred = reg_rh.predict(test_data['img'])

    return lh_test_pred, rh_test_pred

def matrix_add(dict_: dict):
    if dict_ is None or len(dict_) == 0:
        return None
    
    ret = None
    for roi_dict in dict_.values():
        for value in roi_dict.values():
            if ret is None:
                ret = value
            else:
                ret = np.add(ret, value)

    return ret


if __name__ == '__main__':

    for subj in range(1, 9):
        print('subj:'+str(subj)+'\n')
        subj = 'subj'+format(subj, '02')
        
        train_pkl_path = os.path.join('/data/SunYang/datasets/Algonauts_dataset/pkl_clip', subj, "train.pkl")
        val_pkl_path = os.path.join('/data/SunYang/datasets/Algonauts_dataset/pkl_clip', subj, "val.pkl")
        test_pkl_path = os.path.join('/data/SunYang/datasets/Algonauts_dataset/pkl_clip', subj, "test.pkl")
        if os.path.exists(train_pkl_path) and os.path.exists(val_pkl_path) and os.path.exists(test_pkl_path):
            with open(train_pkl_path, 'rb') as f:
                train_data = pickle.load(f)
            with open(val_pkl_path, 'rb') as f:
                valid_data = pickle.load(f)
            with open(test_pkl_path, 'rb') as f:
                test_data = pickle.load(f)
        else:
            train_data, valid_data, test_data = consolidate_subj(subj)

        
        hemisphere_list = [hemisphere_left[0]+'h', hemisphere_right[0]+'h']
        # , 'floc-faces', 'floc-places', 'floc-words', 'streams'
        roi_class_list = ['prf-visualrois', 'floc-bodies']
        
        test_pred_dict = {}
        roi_class_roi_map = {} # roi类别的roi映射 key:roi_class value:{roi: map}
        for roi_class in roi_class_list:
            roi_map_dir = os.path.join(data_dir, subj, 'roi_masks', 'mapping_'+roi_class+'.npy')
            roi_map = np.load(roi_map_dir, allow_pickle=True).item()
            # 删掉Unknown
            roi_map.pop(0)
            
            # 生成roi类别的roi映射
            roi_class_roi_map[roi_class] = {v: k for k, v in map(lambda x: (x[0], x[1]), roi_map.items())}

            # 生成 最终结果字典 结构
            for hemisphere in hemisphere_list:
                dict_ = {v: None for v in roi_map.values()} # roi作为key
                if test_pred_dict.get(hemisphere) is None:
                    test_pred_dict[hemisphere] = {roi_class: dict_}
                else:
                    test_pred_dict[hemisphere][roi_class] = dict_
        
        for roi_class, roi_dict in test_pred_dict[hemisphere_list[0]].items():
            for roi in roi_dict.keys():
                roi_mapping = roi_class_roi_map[roi_class][roi]
                test_pred_dict[hemisphere_list[0]][roi_class][roi], \
                test_pred_dict[hemisphere_list[1]][roi_class][roi] = \
                linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, roi_mapping)

        # 汇总test预测结果 lh
        lh_fmri_test_pred = matrix_add(test_pred_dict[hemisphere_list[0]])
        # 汇总test预测结果 rh
        rh_fmri_test_pred = matrix_add(test_pred_dict[hemisphere_list[1]])

        # roi_class prf-visualrois
        roi_class = 'prf-visualrois'
        roi_map_dir = os.path.join(data_dir, subj, 'roi_masks', 'mapping_'+roi_class+'.npy')
        roi_map = np.load(roi_map_dir, allow_pickle=True).item()
        for roi in roi_map.values():
            if roi == 'Unknown':
                continue
            roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]
        
        
        lh_fmri_test_pred_V1v, rh_fmri_test_pred_V1v = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'V1v')
        lh_fmri_test_pred_V1d, rh_fmri_test_pred_V1d = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'V1d')
        lh_fmri_test_pred_V2v, rh_fmri_test_pred_V2v = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'V2v')
        lh_fmri_test_pred_V2d, rh_fmri_test_pred_V2d = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'V2d')
        lh_fmri_test_pred_V3v, rh_fmri_test_pred_V3v = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'V3v')
        lh_fmri_test_pred_V3d, rh_fmri_test_pred_V3d = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'V3d')
        lh_fmri_test_pred_hV4, rh_fmri_test_pred_hV4 = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'hV4')
        lh_fmri_prf_visualrois_test_pred = matrix_add(lh_fmri_test_pred_V1v, lh_fmri_test_pred_V1d, 
                                                      lh_fmri_test_pred_V2v, lh_fmri_test_pred_V2d, 
                                                      lh_fmri_test_pred_V3v, lh_fmri_test_pred_V3d, 
                                                      lh_fmri_test_pred_hV4)
        rh_fmri_prf_visualrois_test_pred = matrix_add(rh_fmri_test_pred_V1v, rh_fmri_test_pred_V1d, 
                                                      rh_fmri_test_pred_V2v, rh_fmri_test_pred_V2d, 
                                                      rh_fmri_test_pred_V3v, rh_fmri_test_pred_V3d, 
                                                      rh_fmri_test_pred_hV4)
        # roi_class floc-bodies
        roi_class = 'floc-bodies'
        lh_fmri_test_pred_EBA, rh_fmri_test_pred_EBA = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'EBA')
        lh_fmri_test_pred_FBA_1, rh_fmri_test_pred_FBA_1 = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'FBA-1')
        lh_fmri_test_pred_FBA_2, rh_fmri_test_pred_FBA_2 = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'FBA-2')
        lh_fmri_test_pred_mTL_bodies, rh_fmri_test_pred_mTL_bodies = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'mTL-bodies')
        lh_fmri_floc_bodies_test_pred = matrix_add(lh_fmri_test_pred_EBA, lh_fmri_test_pred_FBA_1, 
                                                      lh_fmri_test_pred_FBA_2, lh_fmri_test_pred_mTL_bodies)
        rh_fmri_floc_bodies_test_pred = matrix_add(rh_fmri_test_pred_EBA, rh_fmri_test_pred_FBA_1, 
                                                      rh_fmri_test_pred_FBA_2, rh_fmri_test_pred_mTL_bodies)
        # roi_class floc-faces
        roi_class = 'floc-faces'
        lh_fmri_test_pred_OFA, rh_fmri_test_pred_OFA = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'OFA')
        lh_fmri_test_pred_FFA_1, rh_fmri_test_pred_FFA_1 = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'FFA-1')
        lh_fmri_test_pred_FFA_2, rh_fmri_test_pred_FFA_2 = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'FFA-2')
        lh_fmri_test_pred_mTL_faces, rh_fmri_test_pred_mTL_faces = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'mTL-faces')
        lh_fmri_test_pred_aTL_faces, rh_fmri_test_pred_aTL_faces = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'aTL-faces')
        lh_fmri_floc_faces_test_pred = matrix_add(lh_fmri_test_pred_OFA, lh_fmri_test_pred_FFA_1, 
                                                      lh_fmri_test_pred_FFA_2, lh_fmri_test_pred_mTL_faces,
                                                      lh_fmri_test_pred_aTL_faces)
        rh_fmri_floc_faces_test_pred = matrix_add(rh_fmri_test_pred_OFA, rh_fmri_test_pred_FFA_1, 
                                                      rh_fmri_test_pred_FFA_2, rh_fmri_test_pred_mTL_faces,
                                                      rh_fmri_test_pred_aTL_faces)
        # roi_class floc-places
        roi_class = 'floc-places'
        lh_fmri_test_pred_OPA, rh_fmri_test_pred_OPA = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'OPA')
        lh_fmri_test_pred_PPA, rh_fmri_test_pred_PPA = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'PPA')
        lh_fmri_test_pred_RSC, rh_fmri_test_pred_RSC = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'RSC')
        lh_fmri_floc_places_test_pred = matrix_add(lh_fmri_test_pred_OPA, lh_fmri_test_pred_PPA, 
                                                      lh_fmri_test_pred_RSC)
        rh_fmri_floc_places_test_pred = matrix_add(rh_fmri_test_pred_OPA, rh_fmri_test_pred_PPA, 
                                                      rh_fmri_test_pred_RSC)
        # roi_class floc-words
        roi_class = 'floc-words'
        lh_fmri_test_pred_OWFA, rh_fmri_test_pred_OWFA = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'OWFA')
        lh_fmri_test_pred_VWFA_1, rh_fmri_test_pred_VWFA_1 = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'VWFA-1')
        lh_fmri_test_pred_VWFA_2, rh_fmri_test_pred_VWFA_2 = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'VWFA-2')
        lh_fmri_test_pred_mfs_words, rh_fmri_test_pred_mfs_words = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'mfs-words')
        lh_fmri_test_pred_mTL_words, rh_fmri_test_pred_mTL_words = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'mTL-words')
        lh_fmri_floc_words_test_pred = matrix_add(lh_fmri_test_pred_OWFA, lh_fmri_test_pred_VWFA_1, 
                                                      lh_fmri_test_pred_VWFA_2, lh_fmri_test_pred_mfs_words,
                                                      lh_fmri_test_pred_mTL_words)
        rh_fmri_floc_words_test_pred = matrix_add(rh_fmri_test_pred_OWFA, rh_fmri_test_pred_VWFA_1, 
                                                      rh_fmri_test_pred_VWFA_2, rh_fmri_test_pred_mfs_words, 
                                                      rh_fmri_test_pred_mTL_words)
        # roi_class streams
        roi_class = 'streams'
        lh_fmri_test_pred_early, rh_fmri_test_pred_early = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'early')
        lh_fmri_test_pred_midventral, rh_fmri_test_pred_midventral = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'midventral')
        lh_fmri_test_pred_midlateral, rh_fmri_test_pred_midlateral = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'midlateral')
        lh_fmri_test_pred_midparietal, rh_fmri_test_pred_midparietal = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'midparietal')
        lh_fmri_test_pred_ventral, rh_fmri_test_pred_ventral = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'ventral')
        lh_fmri_test_pred_lateral, rh_fmri_test_pred_lateral = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'lateral')
        lh_fmri_test_pred_parietal, rh_fmri_test_pred_parietal = linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, 'parietal')
        lh_fmri_streams_test_pred = matrix_add(lh_fmri_test_pred_early, lh_fmri_test_pred_midventral, 
                                                      lh_fmri_test_pred_midlateral, lh_fmri_test_pred_midparietal,
                                                      lh_fmri_test_pred_ventral, lh_fmri_test_pred_lateral, 
                                                      lh_fmri_test_pred_parietal)
        rh_fmri_streams_test_pred = matrix_add(rh_fmri_test_pred_early, rh_fmri_test_pred_midventral, 
                                                      rh_fmri_test_pred_midlateral, rh_fmri_test_pred_midparietal, 
                                                      rh_fmri_test_pred_ventral, rh_fmri_test_pred_lateral, 
                                                      rh_fmri_test_pred_parietal)

        # 汇总test预测结果 lh
        lh_fmri_test_pred = matrix_add(lh_fmri_prf_visualrois_test_pred, lh_fmri_floc_bodies_test_pred, 
                                       lh_fmri_floc_faces_test_pred, lh_fmri_floc_places_test_pred, 
                                       lh_fmri_floc_words_test_pred, lh_fmri_streams_test_pred)
        # 汇总test预测结果 rh
        rh_fmri_test_pred = matrix_add(rh_fmri_prf_visualrois_test_pred, rh_fmri_floc_bodies_test_pred, 
                                       rh_fmri_floc_faces_test_pred, rh_fmri_floc_places_test_pred, 
                                       rh_fmri_floc_words_test_pred, rh_fmri_streams_test_pred)

        # 保存测试结果，用于上传
        lh_fmri_test_pred = lh_fmri_test_pred.astype(np.float32)
        rh_fmri_test_pred = rh_fmri_test_pred.astype(np.float32)
        subject_submission_dir = os.path.join(parent_submission_dir,'subj'+format(subj, '02'))
        if os.path.exists(subject_submission_dir) is False:
            os.mkdir(subject_submission_dir)
        np.save(os.path.join(subject_submission_dir, 'lh_pred_test.npy'), lh_fmri_test_pred)
        np.save(os.path.join(subject_submission_dir, 'rh_pred_test.npy'), rh_fmri_test_pred)

