import torch


def apply_model(model, dataset, batch_size=100, device='cpu'):
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0
    )
    X, Y = [], []

    model.eval()
    model.to(device)

    with torch.no_grad():
        for x, y in dl:

            x = x.to(device)
            x = model(x)

            # If the model returns more than one tensor, e.g., encoder of an
            # autoencoder model, take the first one as output...
            if isinstance(x, (tuple, list)):
                x = x[0]

            X += x.cpu().tolist()
            Y += (y.tolist())

    return X, Y


def argmax_and_accuracy(X, Y, binary=False):

    if not binary:
        decision = torch.tensor(X).max(1)[1]
    else:
        X = torch.tensor(X).squeeze()
        assert X.ndimension() == 1
        decision = (X > 0.0).long()

    Y = torch.tensor(Y)

    correct = (decision == Y).sum().float().item()
    num_samples = float(Y.size(0))
    return (correct/num_samples)*100.0
