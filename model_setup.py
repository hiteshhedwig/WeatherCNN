import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import config
from dataloader import load_dataset, split_into_trainvals
from train import train_model


def fine_tune_model(model,class_names=5):

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, class_names)
    model_ft = model.to("cpu")

    return model_ft


if __name__ =="__main__":

    data_dir = "/content/gdrive/MyDrive/Intern DL/dataset"

    dataloaders, image_datasets, class_names= load_dataset(data_dir)
    train_loader, validation_loader, dataset_sizes= split_into_trainvals(dataloaders, image_datasets)
    model = models.resnet18(pretrained=True)
    model_ft= fine_tune_model(model)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_trained = train_model(model_ft, criterion, 
                optimizer_ft, exp_lr_scheduler, 
                train_loader,validation_loader,
                dataset_sizes,
                num_epochs=5)

    #saving model
    PATH= '/content/gdrive/MyDrive/Intern DL/model_internship.pth'
    torch.save(model_trained.state_dict(), PATH)

    
