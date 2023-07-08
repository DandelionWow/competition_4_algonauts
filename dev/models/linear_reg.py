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
