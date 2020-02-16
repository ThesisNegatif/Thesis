import argparse
from time import time, sleep
from os import listdir
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import json
from workspace_utils import active_session
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from collections import OrderedDict
import nutils
from nutils import transform_data, model_setup, save_checkpoint

def main():
    in_arg = get_input_args()
    trainloader, validloader, testloader, class_to_indx = transform_data(in_arg.data_dir)
    model, criterion, optimizer = model_setup(in_arg.arch, in_arg.dropout, in_arg.hidden_units, in_arg.learning_rate)
    trained_model = train_validate_classifier(model, trainloader, validloader, criterion, optimizer, in_arg.epochs, in_arg.gpu)
    check_accuracy_on_test(trained_model, testloader, criterion, in_arg.gpu)
    save_checkpoint(trained_model, in_arg.epochs, in_arg.save_dir, in_arg.arch, in_arg.learning_rate, class_to_indx, optimizer,                                       in_arg.hidden_units, in_arg.dropout)
    
    pass
    
    
def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str, default="./flowers/",
                        help='main data directory')
    parser.add_argument('--save_dir', type=str, default='', 
                        help='directory to save files')
    parser.add_argument('--arch', type=str, default='densenet121', 
                        help='chosen model')
    parser.add_argument('--learning_rate', type=float, default='0.001',
                        help='chosen learning rate')
    parser.add_argument('--hidden_units', type=int, default='500',
                        help='chosen hidden units size')
    parser.add_argument('--epochs', type=int, default='12',
                        help='chosen number of epochs')
    parser.add_argument('--gpu', default=False, action='store_true', help='gpu or cpu')
    parser.add_argument('--dropout', type=float, default='0.5',
                        help='dropout amount')

    return parser.parse_args()


def train_validate_classifier(model, trainloader, validloader, criterion, optimizer, epochs, gpu):
    print_every = 40
    steps = 0
    running_loss = 0

    if gpu and torch.cuda.is_available():
        model.cuda()
    elif gpu and torch.cuda.is_available() == False:
        print('GPU selected but drivers not found, training with CPU')
    else:
        print('Training with CPU')

    for e in range(epochs):
        model.train()
        
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = Variable(inputs), Variable(labels)
            if gpu and torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                accuracy = 0
                validation_loss = 0
                
                for ii, (inputs2, labels2) in enumerate(validloader):
                    optimizer.zero_grad()
                    inputs2, labels2 = Variable(inputs2), Variable(labels2)
                    if gpu and torch.cuda.is_available():
                        inputs2, labels2 = inputs2.cuda(), labels2.cuda()
                    with torch.no_grad():
                        outputs = model.forward(inputs2)
                        validation_loss += criterion(outputs, labels2).data.item()
                        
                        probabilities = torch.exp(outputs).data
                        equality = (labels2.data == probabilities.max(1)[1])
                        
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.4f}".format(validation_loss/len(validloader)),
                      "Validation Accuracy: {:.4f}".format(accuracy/len(validloader)))
                running_loss = 0
                model.train()
    
    return model

def check_accuracy_on_test(model, testloader, criterion, gpu):
    if gpu and torch.cuda.is_available():
        model.cuda()
    elif gpu and torch.cuda.is_available() == False:
        print('GPU selected but drivers not found, training with CPU')
    else:
        print('Training with CPU')
    model.eval()
    test_loss = 0
    accuracy = 0                  
    for inputs, labels in iter(testloader):
        inputs, labels = Variable(inputs), Variable(labels)
        if gpu and torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()              
        with torch.no_grad():              
            outputs = model.forward(inputs)              
            test_loss += criterion(outputs, labels).data.item() 
            probabilities = torch.exp(outputs).data
            equality = (labels.data == probabilities.max(1)[1])          
            accuracy += equality.type_as(torch.FloatTensor()).mean()          
    print("Test Loss: {:.4f}.. ".format(test_loss/len(testloader)),
          "Test Accuracy: {:.4f}".format(accuracy/len(testloader)))   
    
    pass                      
    

if __name__ == "__main__":
    main()

