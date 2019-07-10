import torch


def apply_model(model, dataset, batch_size=100, device='cpu'):
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=5
    )
    X, Y = [], []

    model.eval()
    model.to(device)

    with torch.no_grad():
        for x, y in dl:

            x = x.to(device)
            x = model(x)

            X += x.cpu().tolist()
            Y += (y.tolist())

    return X, Y
