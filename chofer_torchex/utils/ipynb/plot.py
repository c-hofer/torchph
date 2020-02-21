import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


def plt_img_grid(images, nrow=8, figure=None, scale_each=True):
    """
    Plot images on a grid.
    """
    if figure is None:
        plt.figure(figsize=(16, 16))

    ax = plt.gca()
    img = make_grid(images, normalize=True, nrow=nrow, scale_each=scale_each)
    npimg = img.cpu().detach().numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
