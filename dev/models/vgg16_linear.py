'''
使用预训练的vgg16作为特征提取
'''
import torch.nn as nn
from torchvision.models import vgg16

def get_model(out_feature_dim):
    model = vgg16(pretrained=True)
    model.classifier[-1] = nn.Linear(4096, out_feature_dim)
    return model
