'''
使用预训练的alexnet作为特征提取
'''
import torch.nn as nn
from torchvision.models import alexnet

def get_model(out_feature_dim):
    model = alexnet(pretrained=True)
    model.classifier[-1] = nn.Linear(4096, out_feature_dim)
    return model
