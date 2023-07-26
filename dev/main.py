# main.py: define a main function to run the whole pipeline

import os
import torch
import config
from dataloader import (
    create_data_loader, 
    create_data_loader_4_cnn, 
    create_test_data_loader, 
    create_test_data_loader_4_cnn,
    create_test_data_loader_4_resnet,
    create_test_data_loader_4_vgg,
    create_data_loader_4_resnet,
    create_data_loader_4_vgg,
    create_data_loader_4_clip,
    create_test_data_loader_4_clip,
)
from models import vgg16_linear
from models import alexnet_linear
from models import resnet_linear
from models.vgg16_mlp import Vgg16MLPModel
from models.clip_linear import ClipLinearModel
from models.cnn_linear import CNNModel
from models.linear_reg import LinearRegression, LinearRegression3Layer
from loss import create_criterion_and_optimizer
from train import train
from test import test
from valid import valid
import numpy as np

_STR_TRAIN = 'train'
_STR_TEST = 'test'
_STR_VAL = 'val'
_STR_HEMISPHERE_LEFT = 'lh'
_STR_HEMISPHERE_RIGHT = 'rh'

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

def get_roi_idx_dict(data_dir, subj, hemisphere_list, roi_class_roi_map, device):
    roi_idx_dict = {}
    # 遍历左右脑
    for hemisphere in hemisphere_list:
        roi_idx_dict[hemisphere] = {}
        # 遍历所有roi_class
        for roi_class, roi_map in roi_class_roi_map.items():
            # 加载challenge_space
            challenge_roi_class_dir = os.path.join(data_dir, subj, 'roi_masks', 
                                                hemisphere+'.'+roi_class+'_challenge_space.npy')
            challenge_roi_class = np.load(challenge_roi_class_dir)
            roi_idx_dict[hemisphere][roi_class] = {}
            # 遍历某个roi_class的所有roi_map
            for roi, roi_mapping in roi_map.items():
                # 在challenge_space中找出对应roi的索引
                fmri_roi_idx = np.where(challenge_roi_class == roi_mapping)[0]
                fmri_roi_idx = torch.tensor(fmri_roi_idx, dtype=torch.int64).to(device)
                roi_idx_dict[hemisphere][roi_class][roi] = fmri_roi_idx
    
    return roi_idx_dict

def summary_test_pred(subject_submission_dir, test_pred_dict: dict, roi_idx_dict: dict, hemisphere, device):
    pred_dict = test_pred_dict[hemisphere]
    if pred_dict is None or len(pred_dict) == 0:
        return None
    
    ret = np.load(os.path.join(subject_submission_dir, hemisphere+'_pred_test.npy'))
    ret = torch.tensor(ret).to(device)

    for roi_class, roi_dict in pred_dict.items():
        for roi, value in roi_dict.items():
            # 跳过未预测的roi
            if value is None:
                continue

            if roi_class == 'streams':
                # streams用于填充值为0的idx
                ret_zeros_like = torch.zeros(ret.shape[1])
                ret_zeros_idx = torch.where(ret[0, :] == 0)[0]
                ret_zeros_like[ret_zeros_idx] = 1
                roi_idx = roi_idx_dict[hemisphere][roi_class][roi]
                ret_zeros_like[roi_idx] += 1
                ret_fill_idx = torch.where(ret_zeros_like == 2)[0]
                if len(ret_fill_idx) == 0:
                    continue
                # 对应fill_idx，重置value
                ret_zeros_like = torch.zeros_like(ret)
                ret_zeros_like[:, roi_idx] = value
                value = ret_zeros_like[:, ret_fill_idx]
            else:
                # 这里应该考虑index重复的情况把？
                ret_fill_idx = roi_idx_dict[hemisphere][roi_class][roi]

            ret[:, ret_fill_idx] = value
            
    return ret

def save_test_pred(subject_submission_dir, lh_pred_fmri, rh_pred_fmri):
    # 转换
    if torch.is_tensor(lh_pred_fmri):
        lh_pred_fmri = lh_pred_fmri.cpu().numpy()
    if torch.is_tensor(rh_pred_fmri):
        rh_pred_fmri = rh_pred_fmri.cpu().numpy()
    lh_pred_fmri = lh_pred_fmri.astype(np.float32)
    rh_pred_fmri = rh_pred_fmri.astype(np.float32)
    # 保存
    if os.path.exists(subject_submission_dir) is False:
        os.mkdir(subject_submission_dir)
    np.save(os.path.join(subject_submission_dir, "lh_pred_test.npy"), lh_pred_fmri)
    np.save(os.path.join(subject_submission_dir, "rh_pred_test.npy"), rh_pred_fmri)

def save_checkpoint(epoch, model, optimizer, subj, hemisphere, roi, config):
    if config['is_save_checkpoint'] == 1:
        path = os.path.join(config['checkpoint_path'], subj)
        if os.path.exists(path) is False:
            os.mkdir(path)
        path = os.path.join(path, 'model_'+hemisphere+'_'+roi+'.pt')
        
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
            path
        )

def load_checkpoint(model, optimizer, subj, hemisphere, roi, config):
    epoch = 0
    if config['is_load_checkpoint'] == 1:
        path = os.path.join(config['checkpoint_path'], subj, 'model_'+hemisphere+'_'+roi+'.pt')
        
        checkpoint = torch.load(path)

        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return epoch, model, optimizer

def train_model(model, data_loader_dict, roi_idx, roi, hemisphere, subj, device, cfg):
    model.to(device)
    
    # 实例化损失，优化器，学习率更新
    criterion, optimizer, scheduler = create_criterion_and_optimizer(model, cfg)

    # 加载checkpoint
    epoch_, model, optimizer = load_checkpoint(model, optimizer, subj, hemisphere, roi, cfg)

    # 初始化最佳matric
    best_pearson_metric = -1.0
    best_epoch = 0
    # loop over epochs
    for epoch in range(epoch_, cfg["epochs"]): 
        print(f"Epoch {epoch+1}/{cfg['epochs']}")
        print("-" * 10)
        # 训练
        train(model, data_loader_dict[_STR_TRAIN], criterion, optimizer, device, epoch, hemisphere, roi_idx)
        # 在每个epoch结束后，调用scheduler.step()来更新学习率
        # scheduler.step()
        
        # 保存checkpoint
        save_checkpoint(epoch+1, model, optimizer, subj, hemisphere, roi, cfg)

        with torch.no_grad():
            # 验证
            val_corr = valid(model, data_loader_dict[_STR_VAL], criterion, device, hemisphere, roi_idx)
            
            # 保存最佳模型参数
            if val_corr > best_pearson_metric:
                best_epoch = epoch
                best_pearson_metric = val_corr
                print(f'Saving model with highest metric: {best_pearson_metric:.4f}')
                path = os.path.join(cfg['best_checkpoint_path'], subj)
                if os.path.exists(path) is False:
                    os.mkdir(path)
                path = os.path.join(path, hemisphere+'_'+roi+'.pt')
                torch.save(model.state_dict(), path)

        # 若2个epoch未保存最佳，说明过拟合，终止训练
        if epoch - best_epoch == int(cfg['overfitting_max_epoch']):
            break

def test_model(model, data_loader_dict, roi, hemisphere, subj, cfg, device):
    # 加载最佳模型参数
    path = os.path.join(cfg['best_checkpoint_path'], subj, hemisphere+'_'+roi+'.pt')
    model.load_state_dict(torch.load(path))
    
    with torch.no_grad():
        # 获取测试集结果
        return test(model, data_loader_dict[_STR_TEST], device)

def train_and_test(model_dict: dict, data_loader_dict: dict, roi_idx_dict: dict, subj, roi_class, roi, target_roi_list, test_pred_dict, device, cfg):
    if roi in target_roi_list:
        for hemisphere, model in model_dict.items():
            # 训练模型
            train_model(model, data_loader_dict, roi_idx_dict[hemisphere][roi_class][roi], roi, hemisphere, subj, device, cfg)
            # test集预测
            test_pred_dict[hemisphere][roi_class][roi] = \
                test_model(model, data_loader_dict, roi, hemisphere, subj, cfg, device)
    
    return test_pred_dict

def main(cfg: config):
    # set the device
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    for subj in subj_list:
        subj = "subj" + format(subj, "02")
        print(f"{subj}/08")
        print("-" * 40)

        # 创建dataloader，这里应该区分开不同模型（暂不区分）
        train_data_loader, val_data_loader = create_data_loader_4_cnn(cfg, subj)
        test_data_loader = create_test_data_loader_4_cnn(cfg, subj)
        data_loader_dict = {
            _STR_TRAIN: train_data_loader,
            _STR_TEST: test_data_loader,
            _STR_VAL: val_data_loader,
        }

        # 获取test集的预测结果字典 和 每个roi类别的roi映射
        test_pred_dict, roi_class_roi_map = get_test_pred_dict_and_roi_map(cfg['dataset_path'], subj, hemisphere_list, roi_class_list)
        # 获取每个roi对应fmri数据中的索引
        roi_idx_dict = get_roi_idx_dict(cfg['dataset_path'], subj, hemisphere_list, roi_class_roi_map, device)
        
        model_dict = {
            _STR_HEMISPHERE_LEFT: None,
            _STR_HEMISPHERE_RIGHT: None,
        }
        
        # 遍历
        for roi_class, roi_dict in test_pred_dict[hemisphere_list[0]].items():
            for roi in roi_dict.keys():
                if roi not in roi_list_all_model:
                    continue
                
                print(f"roi: {roi}")
                print("-" * 20)
                # 加载roi对应fmri中的索引
                lh_roi_idx = roi_idx_dict[hemisphere_list[0]][roi_class][roi]
                rh_roi_idx = roi_idx_dict[hemisphere_list[1]][roi_class][roi]
                    
                # ClipLinearModel, roi_list_4_clip_linear
                model_dict[_STR_HEMISPHERE_LEFT] = ClipLinearModel(device, len(lh_roi_idx))
                model_dict[_STR_HEMISPHERE_RIGHT] = ClipLinearModel(device, len(rh_roi_idx))
                test_pred_dict = train_and_test(model_dict, data_loader_dict, roi_idx_dict, 
                               subj, roi_class, roi, roi_list_4_clip_linear, 
                               test_pred_dict, device, cfg)
                # vgg16_linear, roi_list_4_vgg16_linear
                model_dict[_STR_HEMISPHERE_LEFT] = vgg16_linear.get_model(len(lh_roi_idx))
                model_dict[_STR_HEMISPHERE_RIGHT] = vgg16_linear.get_model(len(rh_roi_idx))
                test_pred_dict = train_and_test(model_dict, data_loader_dict, roi_idx_dict, 
                               subj, roi_class, roi, roi_list_4_vgg16_linear, 
                               test_pred_dict, device, cfg)
                # alexnet_linear, roi_list_4_alexnet_linear
                model_dict[_STR_HEMISPHERE_LEFT] = alexnet_linear.get_model(len(lh_roi_idx))
                model_dict[_STR_HEMISPHERE_RIGHT] = alexnet_linear.get_model(len(rh_roi_idx))
                test_pred_dict = train_and_test(model_dict, data_loader_dict, roi_idx_dict, 
                               subj, roi_class, roi, roi_list_4_alexnet_linear, 
                               test_pred_dict, device, cfg)
                # Vgg16MLPModel, roi_list_4_vgg16_mlp
                model_dict[_STR_HEMISPHERE_LEFT] = Vgg16MLPModel(len(lh_roi_idx))
                model_dict[_STR_HEMISPHERE_RIGHT] = Vgg16MLPModel(len(rh_roi_idx))
                test_pred_dict = train_and_test(model_dict, data_loader_dict, roi_idx_dict, 
                               subj, roi_class, roi, roi_list_4_vgg16_mlp, 
                               test_pred_dict, device, cfg)
                    
                # 每个roi汇总保存一次，先读后写
                subject_submission_dir = os.path.join(cfg['parent_submission_dir'], subj)
                lh_pred_fmri = summary_test_pred(subject_submission_dir, test_pred_dict, roi_idx_dict, hemisphere_list[0], device)
                rh_pred_fmri = summary_test_pred(subject_submission_dir, test_pred_dict, roi_idx_dict, hemisphere_list[1], device)
                # 保存
                save_test_pred(subject_submission_dir, lh_pred_fmri, rh_pred_fmri)


if __name__ == "__main__":
    # 若工作目录更换，这里需要修改
    config_file = os.path.join(os.getcwd(), "config", "config.yaml")
    cfg = config.load_config(config_file)

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
    # 不同roi建模可在此声明roi列表，在for中使用if roi in roi_list判断
    roi_list_4_clip_linear = [
        # 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 
        'EBA', 'FBA-1', 'FBA-2', 'mTL-bodies',
        # 'OFA', 
        'FFA-1', 'FFA-2', 'mTL-faces', 'aTL-faces',
        'OPA', 'PPA', 'RSC',
        # 'OWFA', 
        'VWFA-1', 'VWFA-2', 'mfs-words', 
        # 'mTL-words',
        # 'early', 'midventral', 'midlateral', 'midparietal', 
        'ventral', 'lateral', 'parietal'
    ]
    roi_list_4_vgg16_linear = [
        # 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4',
        # 'EBA', 'FBA-1', 'FBA-2', 'mTL-bodies',
        # 'OFA', 'FFA-1', 'FFA-2', 'mTL-faces', 'aTL-faces',
        # 'OPA', 'PPA', 'RSC', 'OWFA', 
        # 'VWFA-1', 'VWFA-2', 'mfs-words', 'mTL-words',
        # 'early', 'midventral', 'midlateral', 'midparietal', 'ventral', 'lateral', 'parietal'
    ]
    # 试一下alexnet
    roi_list_4_alexnet_linear = [
        # 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4',
        # 'EBA', 'FBA-1', 'FBA-2', 'mTL-bodies',
        # 'OFA', 'FFA-1', 'FFA-2', 'mTL-faces', 'aTL-faces',
        # 'OPA', 'PPA', 'RSC',
        # 'OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'mTL-words',
        # 'early', 'midventral', 'midlateral', 'midparietal', 'ventral', 'lateral', 'parietal'
    ]
    roi_list_4_vgg16_mlp = [
        'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4',
        # 'EBA', 'FBA-1', 'FBA-2', 'mTL-bodies',
        'OFA', 
        # 'FFA-1', 'FFA-2', 'mTL-faces', 'aTL-faces',
        # 'OPA', 'PPA', 'RSC', 
        'OWFA', 
        # 'VWFA-1', 'VWFA-2', 'mfs-words', 
        'mTL-words',
        'early', 'midventral', 'midlateral', 'midparietal', 
        # 'ventral', 'lateral', 'parietal'
    ]
    roi_list_all_model = roi_list_4_clip_linear + roi_list_4_vgg16_linear + \
        roi_list_4_alexnet_linear + roi_list_4_vgg16_mlp
    subj_list = [
        # 1,
        # 2,
        # 3,
        # 4,
        # 5,
        # 6,
        7,
        # 8,
    ]

    main(cfg)
