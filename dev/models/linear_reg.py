# model.py: define a custom model class
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, in_features_dim, out_features_dim):
        # num_classes: the number of output classes
        super(LinearRegression, self).__init__()
        # define a linear layer with input size in_features_dim and output size out_features_dim
        self.linear = nn.Linear(in_features = in_features_dim, out_features = out_features_dim)

    def forward(self, x):
        # apply the linear layer
        x = self.linear(x)
        # return the output tensor of shape (batch_size, 19004)
        return x

class LinearRegression3Layer(nn.Module):
    def __init__(self, in_features_dim, out_features_dim):
        # num_classes: the number of output classes
        super(LinearRegression3Layer, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features_dim, in_features_dim), # 修改输入维度为512
            nn.ReLU(),
            nn.Linear(in_features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_features_dim), # 修改输出维度为10094
        )


    def forward(self, x):
        # apply the linear layer
        x = self.linear_relu_stack(x)
        # return the output tensor
        return x