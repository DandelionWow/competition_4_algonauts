# train.py: define a function to train the model for one epoch
import torch

def train(model, data_loader, criterion, optimizer, device, epoch):
    # model: the model object to train
    # data_loader: the data loader object to load the training data
    # criterion: the loss function to use
    # optimizer: the optimizer to use
    # device: the device to run the model on, either 'cpu' or 'cuda'
    # set the model to training mode
    model.train()
    # initialize the running loss and accuracy
    running_loss = 0.0
    running_acc = 0.0
    # loop over the data batches
    for i, (inputs, labels) in enumerate(data_loader):
        # move the inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass
        outputs = model(inputs)
        # compute the loss
        loss = criterion(outputs, labels)
        # backward pass and optimize
        loss.backward()
        optimizer.step()
        # get the predictions and calculate the accuracy
        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == labels).item() / labels.size(0)
        # update the running loss and accuracy
        running_loss += loss.item()
        running_acc += acc
        # print statistics every 200 batches
        if (i + 1) % 200 == 0:
            print('[%d, %5d] loss: %.3f, acc: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200, running_acc / 200))
            # reset the running loss and accuracy
            running_loss = 0.0
            running_acc = 0.0

