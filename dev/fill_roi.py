import os
import numpy as np
import torch
import torch.nn as nn
import config
from dataloader import (
    create_data_loader_4_cnn,
    create_test_data_loader_4_cnn,
)
from valid import valid
from test import test
from main import get_roi_idx_dict, get_test_pred_dict_and_roi_map
from models import vgg16_linear
from models import alexnet_linear
from models import resnet_linear
from models.vgg16_mlp import Vgg16MLPModel
from models.clip_linear import ClipLinearModel

if __name__ == '__main__':
    # 若工作目录更换，这里需要修改
    config_file = os.path.join(os.getcwd(), 'dev', "config", "config.yaml")
    cfg = config.load_config(config_file)
    dataset_path = cfg['dataset_path']
    parent_submission_dir = cfg['parent_submission_dir']

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
    
    subj = 'subj07'
    
    # 加载submission
    subject_submission_dir = os.path.join(parent_submission_dir, subj)
    lh_submission = np.load(os.path.join(subject_submission_dir, hemisphere_list[0]+'_pred_test.npy'))
    rh_submission = np.load(os.path.join(subject_submission_dir, hemisphere_list[1]+'_pred_test.npy'))
    lh_ret_zeros_like = torch.zeros_like(torch.tensor(lh_submission)).to(cfg['device'])
    rh_ret_zeros_like = torch.zeros_like(torch.tensor(rh_submission)).to(cfg['device'])
    
    # test dataloader
    train_data_loader, val_data_loader = create_data_loader_4_cnn(cfg, subj)
    test_data_loader = create_test_data_loader_4_cnn(cfg, subj)

    roi_class = 'streams'
    _, roi_class_roi_map = get_test_pred_dict_and_roi_map(dataset_path, subj, hemisphere_list, roi_class_list)
    roi_idx_dict = get_roi_idx_dict(dataset_path, subj, hemisphere_list, roi_class_roi_map, 'cpu')
    
    for roi in ['early', 'midventral', 'midlateral', 'midparietal']:
        print(roi+'----------------------')
        lh_roi_idx = roi_idx_dict[hemisphere_list[0]][roi_class][roi]
        rh_roi_idx = roi_idx_dict[hemisphere_list[1]][roi_class][roi]
        
        # 加载模型和参数
        # lh_model = ClipLinearModel(cfg['device'], len(lh_roi_idx))
        # rh_model = ClipLinearModel(cfg['device'], len(rh_roi_idx))
        lh_model = Vgg16MLPModel(len(lh_roi_idx)).to(cfg['device'])
        rh_model = Vgg16MLPModel(len(rh_roi_idx)).to(cfg['device'])
        ckt_path = cfg['best_checkpoint_path']
        lh_ckt = torch.load(os.path.join(ckt_path, subj, 'lh_'+roi+'.pt'), map_location=torch.device(cfg['device']))
        rh_ckt = torch.load(os.path.join(ckt_path, subj, 'rh_'+roi+'.pt'), map_location=torch.device(cfg['device']))
        lh_model.load_state_dict(lh_ckt)
        rh_model.load_state_dict(rh_ckt)
        
        # 验证
        # with torch.no_grad():
        #     val_corr = valid(lh_model, val_data_loader, nn.MSELoss(), cfg['device'], 'lh', lh_roi_idx)
        #     val_corr = valid(rh_model, val_data_loader, nn.MSELoss(), cfg['device'], 'rh', rh_roi_idx)
        # 测试
        with torch.no_grad():
            lh_roi_pred = test(lh_model, test_data_loader, cfg['device'])
            rh_roi_pred = test(rh_model, test_data_loader, cfg['device'])

        lh_ret_zeros_like[:, lh_roi_idx] = lh_roi_pred
        rh_ret_zeros_like[:, rh_roi_idx] = rh_roi_pred
    
    for roi in ['ventral', 'lateral', 'parietal']:
        print(roi+'----------------------')
        lh_roi_idx = roi_idx_dict[hemisphere_list[0]][roi_class][roi]
        rh_roi_idx = roi_idx_dict[hemisphere_list[1]][roi_class][roi]
        
        # 加载模型和参数
        lh_model = ClipLinearModel(cfg['device'], len(lh_roi_idx)).to(cfg['device'])
        rh_model = ClipLinearModel(cfg['device'], len(rh_roi_idx)).to(cfg['device'])
        # lh_model = Vgg16MLPModel(len(lh_roi_idx)).to(cfg['device'])
        # rh_model = Vgg16MLPModel(len(rh_roi_idx)).to(cfg['device'])
        ckt_path = cfg['best_checkpoint_path']
        lh_ckt = torch.load(os.path.join(ckt_path, subj, 'lh_'+roi+'.pt'), map_location=torch.device(cfg['device']))
        rh_ckt = torch.load(os.path.join(ckt_path, subj, 'rh_'+roi+'.pt'), map_location=torch.device(cfg['device']))
        lh_model.load_state_dict(lh_ckt)
        rh_model.load_state_dict(rh_ckt)
        
        # 验证
        # with torch.no_grad():
        #     val_corr = valid(lh_model, val_data_loader, nn.MSELoss(), cfg['device'], 'lh', lh_roi_idx)
        #     val_corr = valid(rh_model, val_data_loader, nn.MSELoss(), cfg['device'], 'rh', rh_roi_idx)
        # 测试
        with torch.no_grad():
            lh_roi_pred = test(lh_model, test_data_loader, cfg['device'])
            rh_roi_pred = test(rh_model, test_data_loader, cfg['device'])

        lh_ret_zeros_like[:, lh_roi_idx] = lh_roi_pred
        rh_ret_zeros_like[:, rh_roi_idx] = rh_roi_pred

    fill_roi_class = roi_class_list[0]
    # subj01: V3v V3d hV4 EBA FBA-1 OFA FFA-1 FFA-2 OPA OWFA VWFA-1 VWFA-2 mfs-words
    # subj02: V3v V3d FFA-1 OPA PPA
    # subj03: V1v V1d V2d V3v EBA OWFA
    # subj04: V1v V2d V3d hV4 FBA-1 FBA-2 OFA OPA PPA
    # subj05: FFA-2 VWFA-2 mfs-words
    # subj06: V1d V2d V3v V3d hV4 OPA VWFA-2 mfs-words
    # subj08: OFA FFA-1 FFA-2 OPA OWFA VWFA-1 VWFA-2 mfs-words
    for fill_roi in [
        # 'V1v', 
        # 'V1d', 
        # 'V2v', 
        # 'V2d', 
        # 'V3v', 
        # 'V3d', 
        # 'hV4',
        # 'EBA', 
        # 'FBA-1', 
        # 'FBA-2', 
        # 'mTL-bodies',
        # 'OFA', 
        # 'FFA-1', 
        # 'FFA-2', 
        # 'mTL-faces', 
        # 'aTL-faces',
        # 'OPA', 
        # 'PPA', 
        # 'RSC',
        # 'OWFA', 
        # 'VWFA-1', 
        # 'VWFA-2', 
        # 'mfs-words', 
        # 'mTL-words',
    ]:
        # 覆盖左脑
        fill_roi_idx = roi_idx_dict['lh'][fill_roi_class][fill_roi]
        val = lh_ret_zeros_like[:, fill_roi_idx]
        lh_submission[:, fill_roi_idx] = val.cpu().numpy()
        # 覆盖右脑
        fill_roi_idx = roi_idx_dict['rh'][fill_roi_class][fill_roi]
        val = rh_ret_zeros_like[:, fill_roi_idx]
        rh_submission[:, fill_roi_idx] = val.cpu().numpy()
    
    # 保存
    print(len(rh_submission))
    np.save(os.path.join(subject_submission_dir, hemisphere_list[0]+'_pred_test.npy'), lh_submission)
    np.save(os.path.join(subject_submission_dir, hemisphere_list[1]+'_pred_test.npy'), rh_submission)
