# test.py: define a function to test the model on the test data

def test(model, data_loader, device):
    # model: the model object to test
    # set the model to evaluation mode
    model.eval()
    # loop over the data batches
    for inputs in data_loader:
        # move the inputs and labels to the device
        inputs = inputs.to(device)
        # forward pass
        outputs = model(inputs)

