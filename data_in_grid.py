
import numpy as np
import torchvision
import matplotlib.pyplot as plt
#import dataloader

def imshow(inp,dataloaders,class_names,title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

    inputs,classes= next(iter(dataloaders['Train']))

        # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    grid=imshow(out, title=[class_names[x] for x in classes])
    return grid


