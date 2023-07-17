'''
使用预训练的vgg16作为图片特征提取
'''
import torch
import torch.nn as nn
import torchvision.models as models

class Vgg16MLPModel(nn.Module):
    def __init__(self, out_feature_dim):
        super(Vgg16MLPModel, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.features = self.vgg16.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.mlp = nn.Linear(512 * 7 * 7, out_feature_dim)

    def forward(self, x):
        y = self.features(x)
        y = self.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.mlp(y)
        return y
