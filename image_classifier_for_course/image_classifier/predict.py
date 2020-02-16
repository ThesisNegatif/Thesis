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
from nutils import process_image, return_checkpoint

def main():
    in_arg = get_input_args()
    done_image = process_image(in_arg.input_img)
    model, architecture, class_to_indx = return_checkpoint(in_arg.checkpoint)
    classes, probs = predict(done_image, model, in_arg.top_k, in_arg.gpu, in_arg.category_names, architecture, class_to_indx)
    print_predictions(classes, probs)
    pass


def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_img', type=str, 
                        help='Image to utilise')
    parser.add_argument('checkpoint', type=str, default='checkpoint.pth', 
                        help='Saved checkpoint to use')
    parser.add_argument('--top_k', type=int, default='1', 
                        help='Top KK most likely classes')
    parser.add_argument('--category_names', type=str, default='', 
                        help='Mapping of categories to classes')
    parser.add_argument('--gpu', default=False, action='store_true',
                        help='gpu or cpu')

    return parser.parse_args()


def predict(image, model, top_k, gpu, category_names, architecture, class_to_indx):
    
    image = image.unsqueeze(0).float()
    image = Variable(image)
    if gpu and torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    else:
        print('Utilising CPU') 
 
    with torch.no_grad():
        output = model.forward(image)
        result = torch.exp(output).data.topk(top_k)
    classes = np.array(result[1][0], dtype=np.int)
    probs = Variable(result[0][0]).data
    
    if len(category_names) > 0:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        mapped_classes = {}
        for k in class_to_indx:
            mapped_classes[cat_to_name[k]] = class_to_indx[k]
        mapped_classes = {v:k for k,v in mapped_classes.items()} 
        classes = [mapped_classes[x] for x in classes]
        probs = list(probs)
    else:
        class_to_indx = {v:k for k,v in class_to_indx.items()}
        classes = [class_to_indx[x] for x in classes]
        probs = list(probs)
    return classes, probs

def print_predictions(classes, probs):
    predictions = list(zip(classes, probs))
    for i in range(len(predictions)):
        print('{} : {:.4%}'.format(predictions[i][0], predictions[i][1]))
    
    pass


if __name__ == "__main__":
    main()