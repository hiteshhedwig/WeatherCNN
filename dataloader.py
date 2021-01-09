
import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import config
import os

def split_into_trainvals(dataloaders, image_datasets):
    dataset_size = len(dataloaders['Train'].dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(config.validation_split * dataset_size))

    if config.shuffle_dataset :
        np.random.seed(config.random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(image_datasets['Train'], batch_size=config.batch_size, 
                                            sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(image_datasets['Train'], batch_size=config.batch_size,
                                                    sampler=valid_sampler)
    
    dataset_sizes= {
    'train': len(train_indices),
    'val' :len(valid_sampler) 
    }

    return train_loader, validation_loader, dataset_sizes
    
    
def load_dataset(data_dir):

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            config.data_transforms[x])
                    for x in ['Train']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['Train']}
    class_names = image_datasets['Train'].classes

    return dataloaders, image_datasets, class_names
 


