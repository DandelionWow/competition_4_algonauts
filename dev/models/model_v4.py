'''
clip提取图像特征，左右脑每个ROI分别做线性回归，最后汇总结果

与v3的结果一致

37.5407612557	
'''

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from sklearn.linear_model import LinearRegression
import clip
import pickle

data_dir = '/data/SunYang/datasets/Algonauts_dataset/algonauts_2023_main/algonauts_2023_challenge_data/'
parent_submission_dir = '/data/SunYang/datasets/Algonauts_dataset/algonauts_2023_main/algonauts_2023_challenge_submission'

hemisphere_left = 'left'
hemisphere_right = 'right'

def preprocess_img(subj, dir):
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    data_dir = '/data/SunYang/datasets/Algonauts_dataset/algonauts_2023_main/algonauts_2023_challenge_data'
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
    if os.path.exists(os.path.join(pkl_path, subj)) is False:
        os.mkdir(os.path.join(pkl_path, subj))
    with open(os.path.join(pkl_path, subj, "train.pkl"), 'wb') as f:
        pickle.dump(train_data, f)

    with open(os.path.join(pkl_path, subj, "val.pkl"), 'wb') as f:
        pickle.dump(valid_data, f)

    with open(os.path.join(pkl_path, subj, "test.pkl"), 'wb') as f:
        pickle.dump(test_data, f)

    return train_data, valid_data, test_data

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

def get_test_pred_dict_and_roi_map(data_dir, subj, hemisphere_list, roi_class_list):
    # 用于保存test集的预测结果形式为：{'lh': {'roi_class': {'roi': [...]}}}
    test_pred_dict = {} 
    # roi类别的roi映射 key:roi_class value:{roi: map}
    roi_class_roi_map = {} 
    
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

    return test_pred_dict, roi_class_roi_map

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

        # 左右脑符号 'lh', 'rh'
        hemisphere_list = [hemisphere_left[0]+'h', hemisphere_right[0]+'h']
        # roi类别列表
        roi_class_list = ['prf-visualrois', 'floc-bodies', 'floc-faces', 'floc-places', 'floc-words', 'streams']
        # 获取test集的预测结果字典 和 每个roi类别的roi映射
        test_pred_dict, roi_class_roi_map = get_test_pred_dict_and_roi_map(data_dir, subj, hemisphere_list, roi_class_list)
        
        # 全部做线性回归
        for roi_class, roi_dict in test_pred_dict[hemisphere_list[0]].items():
            for roi in roi_dict.keys():
                roi_mapping = roi_class_roi_map[roi_class][roi]

                # 线性回归预测
                test_pred_dict[hemisphere_list[0]][roi_class][roi], \
                test_pred_dict[hemisphere_list[1]][roi_class][roi] = \
                linear_fit_and_predict(data_dir, subj, train_data, test_data, roi_class, roi_mapping)

        # 汇总test预测结果 lh
        lh_fmri_test_pred = matrix_add(test_pred_dict[hemisphere_list[0]])
        # 汇总test预测结果 rh
        rh_fmri_test_pred = matrix_add(test_pred_dict[hemisphere_list[1]])

        # 保存测试结果，用于上传
        lh_fmri_test_pred = lh_fmri_test_pred.astype(np.float32)
        rh_fmri_test_pred = rh_fmri_test_pred.astype(np.float32)
        subject_submission_dir = os.path.join(parent_submission_dir, subj)
        if os.path.exists(subject_submission_dir) is False:
            os.mkdir(subject_submission_dir)
        np.save(os.path.join(subject_submission_dir, 'lh_pred_test.npy'), lh_fmri_test_pred)
        np.save(os.path.join(subject_submission_dir, 'rh_pred_test.npy'), rh_fmri_test_pred)

