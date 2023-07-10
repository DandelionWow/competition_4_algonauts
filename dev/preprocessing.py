import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import clip

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

if __name__ == '__main__':

    train_img_dir = os.path.join('training_split', 'training_images')
    test_img_dir  = os.path.join('test_split', 'test_images')
    pkl = '/data/SunYang/datasets/Algonauts_dataset/pkl_clip'

    for subj in range(1, 9):
        subj = 'subj' + format(subj, '02')

        all_img_features = preprocess_img(subj, train_img_dir)
        np.save(os.path.join(pkl, subj, 'train_imgs_feature.npy'), all_img_features)

        all_img_features = preprocess_img(subj, test_img_dir)
        np.save(os.path.join(pkl, subj, 'test_imgs_feature.npy'), all_img_features)