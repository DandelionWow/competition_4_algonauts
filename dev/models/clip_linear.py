'''
使用预训练的clip作为特征提取
'''
import torch
import torch.nn as nn
import clip

class ClipLinearModel(nn.Module):
    def __init__(self, device, out_feature_dim):
        super(ClipLinearModel, self).__init__()
        self.clip, _ = clip.load('ViT-B/32', device)
        self.fc = nn.Linear(self.clip.visual.output_dim, out_feature_dim)

    def forward(self, x):
        with torch.no_grad():
            img_feature = self.clip.encode_image(x)
            img_feature = img_feature.float()
        y = self.fc(img_feature)
        return y


def get_model(device, out_feature_dim):
    model, preprocess = clip.load('ViT-B/32', device)
    # 没有fc层
    model.visual.fc = nn.Linear(model.visual.fc.in_features, out_feature_dim)
    return model
