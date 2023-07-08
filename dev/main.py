# main.py: define a main function to run the whole pipeline

import os
import torch
# import the config module and load the config file 
import config 
from dataloader import create_data_loader, create_test_data_loader
from models.linear_reg import LinearRegression
from loss import criterion, optimizer 
from train import train 
from .test import test 

def main(cfg: config):
    # import the config module and load the config file 
    config_file = os.path.join(os.curdir, 'dev', 'config.yaml')
    cfg = config.load_config(config_file) # 这一行就是使用config.py创建config对象

    # check if cuda is available and set the device accordingly 
    device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
    
    # create a data loader object with the config file 
    train_data_loader, val_data_loader = create_data_loader(cfg) 
    test_data_loader = create_test_data_loader(cfg) 

    # 初始化模型
    img_feature0, lh_fmri0, rh_fmri0 = train_data_loader.dataset.__getitem__(0)
    lh_model = LinearRegression(len(img_feature0), len(lh_fmri0))
    rh_model = LinearRegression(len(img_feature0), len(rh_fmri0))

    # move the model to the device 
    lh_model.to(device)
    rh_model.to(device)
    
    # loop over epochs 
    for epoch in range(cfg['epochs']): 
      print(f"Epoch {epoch+1}/{cfg['epochs']}") 
      print("-"*10) 
      # train for one epoch 
      train(lh_model, train_data_loader, criterion, optimizer, device, epoch) 
      train(rh_model, train_data_loader, criterion, optimizer, device, epoch) 
      # test on the test data 
      test(lh_model, test_data_loader, device) 
      test(rh_model, test_data_loader, device) 

if __name__ == '__main__':
    config_file = 'config.yaml'
    cfg = config.load_config(config_file) 

    main(cfg)
