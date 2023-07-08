# loss.py: define a loss function and an optimizer
import torch.nn as nn
import torch.optim as optim

def create_criterion_and_optimizer(model, config):
    # create a MSE loss function
    criterion = nn.MSELoss()

    # create an Adam optimizer with the given learning rate
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    return criterion, optimizer
