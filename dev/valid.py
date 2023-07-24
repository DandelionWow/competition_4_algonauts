import os
from evaluate import calculate_pearson_metric

def valid(model, data_loader, criterion, device, hemisphere, roi_idx):
    model.eval()
    running_loss = .0
    running_corr = .0
    total_corr = .0

    # loop over the data batches
    for i, (inputs, lh_fmri, rh_fmri) in enumerate(data_loader):
        inputs = inputs.float().to(device)
        if hemisphere == "lh":
            labels = lh_fmri
        else:
            labels = rh_fmri
        # 预测
        outputs = model(inputs)

        # 将对应roi的索引值应用到fmri上，消除非该roi的噪声
        labels = labels[:, roi_idx].to(device)

        # 计算loss
        loss = criterion(outputs, labels)

        # 计算皮尔逊指数
        fmri = labels.cpu().numpy()
        pred_fmri = outputs.cpu().numpy()
        img2fmri_corr = calculate_pearson_metric(fmri, pred_fmri)
        running_loss += loss.item()
        running_corr += img2fmri_corr
        total_corr += img2fmri_corr

        if (i + 1) % 10 == 0:
            print(
                "valid [%5d] loss: %.3f corr: %.3f"
                % (i + 1, running_loss / 10, running_corr / 10)
            )
            running_loss = 0.0
            running_corr = 0.0

    return total_corr / len(data_loader)
