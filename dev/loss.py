# loss.py: define a loss function and an optimizer
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def create_criterion_and_optimizer(model, config):
    # create a MSE loss function
    criterion = nn.MSELoss()

    # create an Adam optimizer with the given learning rate
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 假设你使用StepLR调整策略，每10个epoch，学习率乘以0.5
    scheduler = StepLR(optimizer, step_size=32, gamma=0.5)

    return criterion, optimizer, scheduler
