import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import config
import torch
import glob 
import matplotlib.pyplot as plt
from PIL import Image
import random
from model_setup import fine_tune_model

def predict_(img):
    img_t = config.preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0).to("cpu")
    out= model(batch_t)
    _, index = torch.max(out, 1)
    predicted= class_names[index]
    return predicted

def inference_plot():
    # settings
    h, w = 10, 10        # for raster image
    nrows, ncols = 2, 2  # array of sub-plots
    figsize = [6, 8]     # figure size, inches

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        #print(i)
        #print(shuffled_list[i])
        img = shuffled_list[i]
        img = Image.open(img).convert('RGB')
        output= predict_(img)
        axi.imshow(img)
        #to extract class name from file name
        actual_class=shuffled_list[i].split('/')[-1].split('_')[0] 
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # write row/col indices as axes' title for identification
        print(actual_class)
        if actual_class== 'Cloud':
            actual_class='cloudy'
        if actual_class== 'rain':
            actual_class='rainy'

        if output == actual_class:
            axi.set_title(f"Predicted Class : {output}", color="green")
            axi.set_xlabel(f'Actual Class : {actual_class}',color="green")

        else:
            axi.set_title(f"Predicted Class : {output}", color="red")
            axi.set_xlabel(f'Actual Class : {actual_class}',color="red")

    plt.tight_layout(True)
    plt.show()

if __name__ == '__main__':

    PATH = "/home/username/5. Pytorch/intern DL/Model/model_internship.pth"

    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    
    #checkpoint = torch.load(load_path, map_location=map_location)

    model=models.resnet18(pretrained=True).to("cpu")
    model= fine_tune_model(model)
    model.load_state_dict(torch.load(PATH,map_location='cpu'))
    model.eval()
    class_names=['cloudy','foggy','rainy','shine','sunrise']

    PATH_TEST="/home/username/Downloads/archive(1)/dataset/alien_test"
    image_list=[]
    for name in glob.glob("/home/username/Downloads/archive(1)/dataset/alien_test/*"):
        image_list.append(name)

    shuffled_list = random.sample(image_list, len(image_list))
    inference_plot()
    



