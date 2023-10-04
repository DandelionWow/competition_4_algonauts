'''
使用预训练的resnet作为特征提取
'''
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

def get_model(out_feature_dim):
    # model = resnet18(pretrained=True)
    # model = resnet34(pretrained=True)
    model = resnet50(pretrained=True)
    # model = resnet101(pretrained=True)
    # model = resnet152(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, out_feature_dim)
    return model
