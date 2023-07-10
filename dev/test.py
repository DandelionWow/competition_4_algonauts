# test.py: define a function to test the model on the test data
import torch

def test(model, data_loader, device, challenge_roi):
    # set the model to evaluation mode
    model.eval()
    outputs = None
    # loop over the data batches
    for inputs in data_loader:
        # move the inputs and labels to the device
        inputs = inputs.float().to(device)
        # forward pass
        outputs_ = model(inputs)
        outputs_ = torch.mul(outputs_, challenge_roi)
        outputs = outputs_ if outputs is None else torch.cat([outputs, outputs_])
    
    return outputs

