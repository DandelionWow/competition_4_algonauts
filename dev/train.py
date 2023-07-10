# train.py: define a function to train the model for one epoch

def train(model, data_loader, criterion, optimizer, device, epoch, hemisphere):
    # set the model to training mode
    model.train()
    # initialize the running loss
    running_loss = 0.0
    # loop over the data batches
    for i, (inputs, lh_fmri, rh_fmri) in enumerate(data_loader):
        # move the inputs and labels to the device
        inputs = inputs.float().to(device)
        if hemisphere[0]=='l':
            labels = lh_fmri.to(device)
        else:
            labels = rh_fmri.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass
        outputs = model(inputs)
        # compute the loss
        loss = criterion(outputs, labels)
        # backward pass and optimize
        loss.backward()
        optimizer.step()
        # update the running loss
        running_loss += loss.item()
        # print statistics every 200 batches
        if (i + 1) % 10 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            # reset the running loss
            running_loss = 0.0

        

