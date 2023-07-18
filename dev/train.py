# train.py: define a function to train the model for one epoch
from evaluate import calculate_pearson_metric

def train(model, data_loader, criterion, optimizer, device, epoch, hemisphere, roi_idx):
    # set the model to training mode
    model.train()
    # initialize the running loss
    running_loss = .0
    running_corr = .0
    
    # loop over the data batches
    for i, (inputs, lh_fmri, rh_fmri) in enumerate(data_loader):
        # move the inputs and labels to the device
        inputs = inputs.float().to(device)
        if hemisphere=='lh':
            labels = lh_fmri
        else:
            labels = rh_fmri
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass
        outputs = model(inputs)
        
        # 将对应roi的索引值应用到fmri上，消除非该roi的噪声
        labels = labels[:, roi_idx].to(device)

        # compute the loss
        loss = criterion(outputs, labels)
        # backward pass and optimize
        loss.backward()
        optimizer.step()
        # update the running loss
        running_loss += loss.item()
        
        # 计算皮尔逊指数
        fmri = labels.cpu().numpy()
        pred_fmri = outputs.cpu().detach().numpy()
        img2fmri_corr = calculate_pearson_metric(fmri, pred_fmri)
        running_corr += img2fmri_corr
        
        # print statistics every 10 batches
        if (i + 1) % 10 == 0:
            print('[%d, %5d] loss: %.3f corr: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10, running_corr / 10))
            # reset the running loss
            running_loss = 0.0
            running_corr = 0.0

        

