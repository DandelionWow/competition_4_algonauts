import torch

def valid(model, data_loader, criterion, device, hemisphere, roi_idx):
    model.eval()
    running_loss = 0.0

    # loop over the data batches
    for i, (inputs, lh_fmri, rh_fmri) in enumerate(data_loader):
        inputs = inputs.float().to(device)
        if hemisphere=='lh':
            labels = lh_fmri.to(device)
        else:
            labels = rh_fmri.to(device)

        outputs = model(inputs)
        # 将对应roi的索引值应用到fmri上，消除非该roi的噪声
        labels = labels[:, roi_idx]
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            print('valid [%5d] loss: %.3f' % (i + 1, running_loss / 10))
            running_loss = 0.0

        

