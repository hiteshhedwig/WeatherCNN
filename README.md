# DL Intern:
This repository my solution for internship task

# Overview of problem
Show use the use of transfer learning to train a model on less data (<1000 images) and still getting good results (>80% accuracy) to demonstrate use of transfer learning. You are free to pick a model of your choice and dataset of your choice. (just don’t pick dataset which is already used in initial training).

For this problem tools used are:
- Pytorch
- MatplotLib
- Numpy

Dataset used :
- Multiclass weather dataset: https://www.kaggle.com/vijaygiitk/multiclass-weather-dataset
- Classes are : cloudy,foggy,rainy,shine,sunrise.
- Total files were 1500. As per the task, i made the dataset of 850 out of 1000 images with 5 classes. 
Check here: https://drive.google.com/drive/folders/19G5O1wd7YJyUo--c6934JyayAAYy5cH7?usp=sharing

# Task solution:
For easy understanding you can check notebook ______. With easy explaination.

## Split dataset into Train and Validation:
Validation split taken is 0.1
 Reason for that is, our dataset is already small. We want to make sure there remains enough data to train
 So,
 - Before split -> 850 images
 - After split-> 
      - Train: 765 images 
      - Validation : 85 images
 
## Dataset images:
Let's use torchvision's grid to see what type of images we have in dataset.
[img](asset/DL1.png)

## Data Evaluation criterion:
Dataset is slightly balanced. So, we can use accuracy as an evaluation term. Otherwise, we could also go for confusion matrix and other things. That we will not be doing in this.

## Finetuning & Training:
Model chosen for the training is : RESNET-18
We finetune, meaning we change the last dense layer like this. `model_ft.fc = nn.Linear(num_ftrs, 5)`. 5 is number of classes.

Since, its multiclass image classification. We can use : 
- `nn.CrossEntropyLoss()` as a loss function. 
- `SGD(model_ft.parameters(), lr=0.001, momentum=0.9)` as an optimizer.
- `StepLR(optimizer_ft, step_size=7, gamma=0.1)` as a scheduler. That will decay LR by a factor of 0.1 every 7 epochs. 

Using train.py script to train the model: 
`train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=5)`
 We are going to train for the 5 epochs if there is need of more we can increase. Training goes like this :
 ```
 Epoch 0/4
----------
train Loss: 0.1411 Acc: 0.9569
val Loss: 0.1891 Acc: 0.9294

Epoch 1/4
----------
train Loss: 0.1490 Acc: 0.9477
val Loss: 0.1550 Acc: 0.9529

Epoch 2/4
----------
train Loss: 0.1357 Acc: 0.9621
val Loss: 0.0915 Acc: 0.9765

Epoch 3/4
----------
train Loss: 0.1303 Acc: 0.9595
val Loss: 0.2026 Acc: 0.9294

Epoch 4/4
----------
train Loss: 0.1067 Acc: 0.9699
val Loss: 0.1566 Acc: 0.9529

Training complete in 0m 47s
Best val Acc: 0.976471
```
**We can see that even with less than 800 images. Fine tuning pretrained RESNET-18 model works out to be incredibly well. 97% Accuracy**

After that, we save the model. To reuse it in inference!

