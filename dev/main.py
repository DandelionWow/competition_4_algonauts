# main.py: define a main function to run the whole pipeline

import os
import torch
import config
from dataloader import create_data_loader, create_test_data_loader
from models.linear_reg import LinearRegression
from loss import create_criterion_and_optimizer
from train import train
from test import test
import numpy as np

def get_test_pred_dict_and_roi_map(data_dir, subj, hemisphere_list, roi_class_list):
    # 用于保存test集的预测结果形式为：{'lh': {'roi_class': {'roi': [...]}}}
    test_pred_dict = {}
    # roi类别的roi映射 key:roi_class value:{roi: map}
    roi_class_roi_map = {}

    for roi_class in roi_class_list:
        roi_map_dir = os.path.join(
            data_dir, subj, "roi_masks", "mapping_" + roi_class + ".npy"
        )
        roi_map = np.load(roi_map_dir, allow_pickle=True).item()
        # 删掉Unknown
        roi_map.pop(0)

        # 生成roi类别的roi映射
        roi_class_roi_map[roi_class] = {
            v: k for k, v in map(lambda x: (x[0], x[1]), roi_map.items())
        }

        # 生成 最终结果字典 结构
        for hemisphere in hemisphere_list:
            dict_ = {v: None for v in roi_map.values()}  # roi作为key
            if test_pred_dict.get(hemisphere) is None:
                test_pred_dict[hemisphere] = {roi_class: dict_}
            else:
                test_pred_dict[hemisphere][roi_class] = dict_

    return test_pred_dict, roi_class_roi_map

def get_challenge_roi(data_dir, subj, hemisphere, roi_class, roi_mapping, device):
    # 在challenge_space中划分出roi的点
    challenge_roi_class_dir = os.path.join(data_dir, subj, 'roi_masks', 
                                           hemisphere+'.'+roi_class+'_challenge_space.npy')
    challenge_roi_class = np.load(challenge_roi_class_dir)
    challenge_roi = np.asarray(challenge_roi_class == roi_mapping, dtype=int)
    challenge_roi = torch.tensor(challenge_roi, dtype=torch.int8).to(device)

    return challenge_roi

def matrix_add(dict_: dict):
    if dict_ is None or len(dict_) == 0:
        return None
    
    ret = None
    for roi_dict in dict_.values():
        for value in roi_dict.values():
            if ret is None:
                ret = value
            else:
                ret = torch.add(ret, value)

    return ret

def save_test_pred(parent_submission_dir, subj, lh_pred_fmri, rh_pred_fmri):
    if torch.is_tensor(lh_pred_fmri):
        lh_pred_fmri = lh_pred_fmri.cpu().numpy()
    if torch.is_tensor(rh_pred_fmri):
        rh_pred_fmri = rh_pred_fmri.cpu().numpy()
    lh_pred_fmri = lh_pred_fmri.astype(np.float32)
    rh_pred_fmri = rh_pred_fmri.astype(np.float32)
    subject_submission_dir = os.path.join(parent_submission_dir, subj)
    if os.path.exists(subject_submission_dir) is False:
        os.mkdir(subject_submission_dir)
    np.save(os.path.join(subject_submission_dir, "lh_pred_test.npy"), lh_pred_fmri)
    np.save(os.path.join(subject_submission_dir, "rh_pred_test.npy"), rh_pred_fmri)


def main(cfg: config):
    # check if cuda is available and set the device accordingly
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    for subj in range(1, 9):
        print(f"subj {subj}/{8}")
        print("-" * 20)

        subj = "subj" + format(subj, "02")

        # create a data loader object with the config file
        train_data_loader, val_data_loader = create_data_loader(cfg, subj)
        test_data_loader = create_test_data_loader(cfg, subj)

        # 获取test集的预测结果字典 和 每个roi类别的roi映射
        test_pred_dict, roi_class_roi_map = get_test_pred_dict_and_roi_map(data_dir, subj, hemisphere_list, roi_class_list)
        # 全部做线性回归
        for roi_class, roi_dict in test_pred_dict[hemisphere_list[0]].items():
            for roi in roi_dict.keys():
                roi_mapping = roi_class_roi_map[roi_class][roi]
        
                lh_challenge_roi = get_challenge_roi(data_dir, subj, hemisphere_list[0], roi_class, roi_mapping, device)
                rh_challenge_roi = get_challenge_roi(data_dir, subj, hemisphere_list[1], roi_class, roi_mapping, device)

                # 初始化模型
                img_feature0, lh_fmri0, rh_fmri0 = train_data_loader.dataset.__getitem__(0)
                lh_model = LinearRegression(len(img_feature0), len(lh_fmri0))
                rh_model = LinearRegression(len(img_feature0), len(rh_fmri0))

                # move the model to the device
                lh_model.to(device)
                rh_model.to(device)

                # create a criterion and optimizer object with the model and config file
                lh_criterion, lh_optimizer, lh_scheduler = create_criterion_and_optimizer(
                    lh_model, cfg
                )
                rh_criterion, rh_optimizer, rh_scheduler = create_criterion_and_optimizer(
                    rh_model, cfg
                )

                # loop over epochs
                for epoch in range(cfg["epochs"]):
                    print(f"Epoch {epoch+1}/{cfg['epochs']}")
                    print("-" * 10)
                    # train for one epoch
                    train(
                        lh_model,
                        train_data_loader,
                        lh_criterion,
                        lh_optimizer,
                        device,
                        epoch,
                        hemisphere_list[0],
                        lh_challenge_roi,
                    )
                    train(
                        rh_model,
                        train_data_loader,
                        rh_criterion,
                        rh_optimizer,
                        device,
                        epoch,
                        hemisphere_list[1],
                        rh_challenge_roi,
                    )
                    # 在每个epoch结束后，调用scheduler.step()来更新学习率
                    lh_scheduler.step()
                    rh_scheduler.step()

                with torch.no_grad():
                    # 获取测试集结果（roi）
                    test_pred_dict[hemisphere_list[0]][roi_class][roi] = test(lh_model, test_data_loader, device, lh_challenge_roi)
                    test_pred_dict[hemisphere_list[1]][roi_class][roi] = test(rh_model, test_data_loader, device, rh_challenge_roi)
        
        # 汇总结果
        lh_pred_fmri = matrix_add(test_pred_dict[hemisphere_list[0]])
        rh_pred_fmri = matrix_add(test_pred_dict[hemisphere_list[1]])
        # 保存
        save_test_pred(parent_submission_dir, subj, lh_pred_fmri, rh_pred_fmri)


if __name__ == "__main__":
    # 若工作目录更换，这里需要修改
    config_file = os.path.join(os.getcwd(), "config", "config.yaml")
    cfg = config.load_config(config_file)

    data_dir = "/data/SunYang/datasets/Algonauts_dataset/algonauts_2023_main/algonauts_2023_challenge_data/"
    parent_submission_dir = "/data/SunYang/datasets/Algonauts_dataset/algonauts_2023_main/algonauts_2023_challenge_submission"
    # 左右脑符号 'lh', 'rh'
    hemisphere_list = ["lh", "rh"]
    # roi类别列表
    roi_class_list = [
        "prf-visualrois",
        "floc-bodies",
        "floc-faces",
        "floc-places",
        "floc-words",
        "streams",
    ]

    main(cfg)
