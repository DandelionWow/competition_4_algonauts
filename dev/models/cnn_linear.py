# 导入PyTorch相关的模块
import torch.nn as nn


# 定义一个CNN模型类，继承自nn.Module
class CNNModel(nn.Module):
    # 定义初始化函数，设置模型的参数
    def __init__(self, out_feature_dim):
        # 调用父类的初始化函数
        super(CNNModel, self).__init__()
        # 定义一个卷积层，输入通道为3，输出通道为16，卷积核大小为3，步长为1，填充为1
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        # 定义一个批量归一化层，输入通道为16
        self.bn1 = nn.BatchNorm2d(16)
        # 定义一个激活函数层，使用ReLU函数
        self.relu1 = nn.ReLU()
        # 定义一个最大池化层，池化核大小为2，步长为2
        self.maxpool1 = nn.MaxPool2d(2, 2)
        # 定义一个卷积层，输入通道为16，输出通道为32，卷积核大小为3，步长为1，填充为1
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        # 定义一个批量归一化层，输入通道为32
        self.bn2 = nn.BatchNorm2d(32)
        # 定义一个激活函数层，使用ReLU函数
        self.relu2 = nn.ReLU()
        # 定义一个最大池化层，池化核大小为2，步长为2
        self.maxpool2 = nn.MaxPool2d(2, 2)
        # 定义一个线性层，输入特征维度为32*106*106（由于输入图像大小为425*425*3，经过两次最大池化后变为106*106*32），输出特征维度为num_classes（19004）
        self.fc = nn.Linear(32*106*106, out_feature_dim)

    # 定义前向传播函数，输入x是一张图像（形状为224*224*3），输出y是一个向量（长度为19004）
    def forward(self, x):
        # 将x通过第一个卷积层得到输出x1
        x1 = self.conv1(x)
        # 将x1通过第一个批量归一化层得到输出x2
        x2 = self.bn1(x1)
        # 将x2通过第一个激活函数层得到输出x3
        x3 = self.relu1(x2)
        # 将x3通过第一个最大池化层得到输出x4
        x4 = self.maxpool1(x3)
        # 将x4通过第二个卷积层得到输出x5
        x5 = self.conv2(x4)
        # 将x5通过第二个批量归一化层得到输出x6
        x6 = self.bn2(x5)
        # 将x6通过第二个激活函数层得到输出x7
        x7 = self.relu2(x6)
        # 将x7通过第二个最大池化层得到输出x8
        x8 = self.maxpool2(x7)
        # 将x8展平成一维向量x9（形状为32*106*106）
        x9 = x8.view(-1, 32*106*106)
        # 将x9通过线性层得到输出y（形状为19004）
        y = self.fc(x9)
        # 返回y作为模型的输出
        return y
