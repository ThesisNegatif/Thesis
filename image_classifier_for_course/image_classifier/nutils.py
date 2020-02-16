import argparse
from time import time, sleep
from os import listdir
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from workspace_utils import active_session
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
from collections import OrderedDict
import numpy as np

input_size = {'vgg16':25088,'alexnet':9216,'densenet121':1024}

def transform_data(data_dir= "./flowers"):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    class_to_indx = train_data.class_to_idx
    
    return trainloader, validloader, testloader, class_to_indx

def model_setup(architecture='densenet121', dropout=0.5, hidden_layer_1=500, lr=0.001):
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif architecture == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print('{} architecture not valid. Supported arguments: vgg16, densenet121 or alexnet'.format(architecture))
        
    for param in model.parameters():
        param.requires_grad = False
        
        classifier = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(dropout)),
            ('fc1', nn.Linear(input_size[architecture], hidden_layer_1)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hidden_layer_1, 100)),
            ('relu2',nn.ReLU()),
            ('fc3', nn.Linear(100, 75)),
            ('relu3',nn.ReLU()),
            ('fc4',nn.Linear(75, 102)),
            ('output', nn.LogSoftmax(dim=1))
            ]))
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    return model, criterion, optimizer

def save_checkpoint(model, epochs, save_dir, architecture, lr, class_to_indx, optimizer, hidden_layer_1, dropout):
    saved_checkpoint = {
        'epochs': epochs,
        'architecture': architecture,
        'lr': lr,
        'class_to_idx': class_to_indx,
        'optimizer_state': optimizer.state_dict(),
        'state_dict': model.state_dict(),
        'hidden_layer_1': hidden_layer_1,
        'dropout': dropout}
    
    if len(save_dir) == 0:
        save_path = save_dir + 'checkpoint.pth'
    else:
        save_path = save_dir + '/checkpoint.pth'
    
    torch.save(saved_checkpoint, save_path)
    pass
    
def return_checkpoint(filepath='checkpoint.pth'):
    checkpoint = torch.load(filepath)
    architecture = checkpoint['architecture']
    hidden_layer_1 = checkpoint['hidden_layer_1']
    dropout = checkpoint['dropout']
    lr = checkpoint['lr']
    model,_,_ = model_setup(architecture, dropout, hidden_layer_1, lr)
    class_to_indx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, architecture, class_to_indx

def process_image(image_path):
    
    pil_image = Image.open(image_path)
    pil_image.thumbnail((256,256), Image.ANTIALIAS)
    width, height = pil_image.size
    crop_width = 224
    crop_height = 224
    l = (width - crop_width)/2
    t = (height - crop_height)/2
    r = (width + crop_width)/2
    b = (height + crop_height)/2
    pil_image = pil_image.crop((l,t,r,b))
    
    norm_mean = np.array([0.485,0.456,0.406])
    norm_std = np.array([0.229, 0.224, 0.225])
    np_pil_image = np.array(pil_image, dtype=np.float64)
    np_pil_image = np_pil_image / 255.0
    np_pil_image = (np_pil_image - norm_mean) / norm_std
    
    new_image = np_pil_image.transpose(2,0,1)
#    new_image = torch.from_numpy(new_image)

    return torch.from_numpy(new_image)

    
if __name__ == "__main__":
    print('NUtils Runs fine')   