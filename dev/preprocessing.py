import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import clip

from main import get_roi_idx_dict, get_test_pred_dict_and_roi_map

def preprocess_img(subj, dir):
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    model = model.eval()
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
    return all_img_features

def get_roi_info(roi_class, roi, hemisphere, hemi_len, roi_idx_dict, streams_arr):
    roi_arr = np.zeros(hemi_len)
    roi_arr[roi_idx_dict[hemisphere][roi_class][roi].numpy()] += 1
    roi_equals_streams_arr = np.where((roi_arr == streams_arr) & (roi_arr == 1))[0]

    return len(roi_idx_dict[hemisphere][roi_class][roi]), roi_equals_streams_arr, roi_arr


if __name__ == '__main__':

    # train_img_dir = os.path.join('training_split', 'training_images')
    # test_img_dir  = os.path.join('test_split', 'test_images')
    # pkl = '/data/SunYang/datasets/Algonauts_dataset/pkl_clip'

    # for subj in range(1, 9):
    #     subj = 'subj' + format(subj, '02')

    #     all_img_features = preprocess_img(subj, train_img_dir)
    #     np.save(os.path.join(pkl, subj, 'train_imgs_feature.npy'), all_img_features)

    #     all_img_features = preprocess_img(subj, test_img_dir)
    #     np.save(os.path.join(pkl, subj, 'test_imgs_feature.npy'), all_img_features)

    dataset_path = '/data/SunYang/datasets/Algonauts_dataset/algonauts_2023_main/algonauts_2023_challenge_data'
    hemisphere_list = ["lh", "rh"]
    roi_class_list = [
        'prf-visualrois',
        'floc-bodies',
        'floc-faces',
        'floc-places',
        'floc-words',
        'streams',
    ]
    subj = 'subj02'
    _, roi_class_roi_map = get_test_pred_dict_and_roi_map(dataset_path, subj, hemisphere_list, roi_class_list)
    roi_idx_dict = get_roi_idx_dict(dataset_path, subj, hemisphere_list, roi_class_roi_map, 'cpu')
    print(len(roi_idx_dict))

    # 加载submission
    parent_submission_dir = '/data/SunYang/datasets/Algonauts_dataset/algonauts_2023_main/algonauts_2023_challenge_submission'
    subject_submission_dir = os.path.join(parent_submission_dir, subj)
    ret = np.load(os.path.join(subject_submission_dir, hemisphere_list[0]+'_pred_test.npy'))
    # 重置submission
    # lh = np.zeros((395, 18981)).astype(np.float32)
    # rh = np.zeros((395, 20530)).astype(np.float32)
    # np.save(os.path.join(subject_submission_dir, hemisphere_list[0]+'_pred_test.npy'), lh)
    # np.save(os.path.join(subject_submission_dir, hemisphere_list[1]+'_pred_test.npy'), rh)

    lh_roi_arr = np.zeros(19004)
    rh_roi_arr = np.zeros(20544)
    roi_arr = lh_roi_arr
    hemisphere = hemisphere_list[0]
    for roi_class, dt in roi_idx_dict[hemisphere].items():
        if roi_class == 'streams':
            continue
        for roi, v in dt.items():
            roi_arr[v.numpy()] += 1
    roi_arr_greater_than_1_idx = np.where(roi_arr > 1)[0]
    print(len(roi_arr))

    lh_streams_arr = np.zeros(19004)
    rh_streams_arr = np.zeros(20544)
    streams_arr = rh_streams_arr
    hemisphere = hemisphere_list[1]

    # 使用streams填充
    streams_arr[roi_idx_dict[hemisphere]['streams']['early'].numpy()] += 1
    streams_arr[roi_idx_dict[hemisphere]['streams']['midventral'].numpy()] += 1
    streams_arr[roi_idx_dict[hemisphere]['streams']['midlateral'].numpy()] += 1
    streams_arr[roi_idx_dict[hemisphere]['streams']['midparietal'].numpy()] += 1
    streams_arr[roi_idx_dict[hemisphere]['streams']['ventral'].numpy()] += 1
    streams_arr[roi_idx_dict[hemisphere]['streams']['lateral'].numpy()] += 1
    streams_arr[roi_idx_dict[hemisphere]['streams']['parietal'].numpy()] += 1
    
    # 统计每个roi信息（除streams）
    # prf-visualrois
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('prf-visualrois', 'V1v', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('prf-visualrois', 'V1d', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('prf-visualrois', 'V2v', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('prf-visualrois', 'V2d', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('prf-visualrois', 'V3v', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('prf-visualrois', 'V3d', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('prf-visualrois', 'hV4', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    # floc-words
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('floc-words', 'OWFA', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('floc-words', 'VWFA-1', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('floc-words', 'VWFA-2', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('floc-words', 'mfs-words', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('floc-words', 'mTL-words', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    # floc-places
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('floc-places', 'OPA', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('floc-places', 'PPA', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('floc-places', 'RSC', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    # floc-faces
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('floc-faces', 'OFA', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('floc-faces', 'FFA-1', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('floc-faces', 'FFA-2', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('floc-faces', 'mTL-faces', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('floc-faces', 'aTL-faces', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    # floc-bodies
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('floc-bodies', 'EBA', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('floc-bodies', 'FBA-1', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('floc-bodies', 'FBA-2', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)
    roi_len, roi_equals_streams_arr, roi_arr =\
          get_roi_info('floc-bodies', 'mTL-bodies', hemisphere, len(streams_arr), roi_idx_dict, streams_arr)

    print(len(roi_idx_dict))