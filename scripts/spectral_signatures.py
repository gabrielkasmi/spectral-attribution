#!/usr/bin/env python
# -*- coding: utf-8 -*-

# computes the spectral signatures (ie. set of wcams)
# given a model backbone

# library imports

import sys
sys.path.append('../')

import argparse
import numpy as np
import torchvision
import torch
from torchvision.models import resnet50, vgg16, vit_b_16
from PIL import Image
import os
from spectral_sobol.torch_explainer import WaveletSobol


# args
parser = argparse.ArgumentParser(description = 'Computation of the spectral signatures')

parser.add_argument('--count', default = 5000, help = "total number of samples to generate", type=int)
parser.add_argument('--batch', default = 10, help = "number of samples per (meta)-batch", type=int)
parser.add_argument('--target_dir', default = '../../data/wcams-imagenet', help = "directory where the samples will be stored", type=str)
parser.add_argument('--backbone', default = 'vitb16', help = "name of the backbone under consideration", type=str)
parser.add_argument('--source_dir', default = '../../data/ImageNet', help = "directory where IN validation set is located", type=str)

args = parser.parse_args()

# retrieve the arguments
count = args.count # total number of samples to generate
batch = args.batch # number of samples per (meta)-batch
target = args.target_dir # directory where the samples will be stored
backbone = args.backbone # name of the backbone under consideration
source_dir = args.source_dir # directory where IN validation set is located

# model
if backbone == 'resnet':
    model = resnet50(pretrained =True) 

elif backbone == "vgg":
    model = vgg16(pretrained = True)

elif backbone == 'vitb16':
    model = vit_b_16(pretrained = True)

## add models here

# turn the model to evaluation mode.
model.eval()

# dictionnary with the labels
# labels dictoinnary
# contains all the names of the IN val set images and their labels
f = open(os.path.join(source_dir, "val.txt"), "r")
raw = f.read().split('\n')
items = {r[:28] : int(r[29:]) for r in raw if r[:10] == 'ILSVRC2012'}

# set up the directories
target_dir = os.path.join(target, backbone)

def generate_target_samples(target, count, items, init = "resnet"):
    """
    generates the list of images for which the wcam will be computed

    source_dir : where the IN validation set is located (with additional data)
    target_dir : the target directory
    count : the number of samples to generate

    init (opt) : whether this list should be initialized
                 if false, will look into another directory, passed as a string
                 to retrieve the list of images to compute
    """

    if init == True:
        # in this case, we pick n samples from the dictionnary
        np.random.seed(42)
        initial_list = np.random.choice(list(items.keys()), count).tolist()

    else:
        # otherwise, go look for the images generated for another backbone and take the latter as a reference
        example_dir = os.path.join(target, init)
        initial_list = os.listdir(example_dir)

        print(len(initial_list))

    return initial_list

def update_sample_list(initial_list, completed, source_dir, items, batch):
    """
    updates the samples list from which the wcams should be computed

    args
    - source_dir : path to IN data
    - items : the dictionnary of labels
    - batch : the size of the batch

    returns 
    - sample_list : a tuple with the path to the image and the label
    - image_name : the corresponding list of image names (without the rest of the path)
    """
    if len(completed) == 0: # if we are at the first round and no wcams where generated so far.
        image_names = initial_list[:batch]
        sample_list = [(os.path.join(source_dir, k), items[k]) for k in image_names][:batch]

    else:
        image_names = [i for i in initial_list if not i in completed][:batch]
        sample_list = [(os.path.join(source_dir, k), items[k]) for k in image_names][:batch]

    return sample_list, image_names

def compute_wcams(sample_list, model):
    """
    generates the wcams

    args
    - sample_list : a list of tuples where 
                                    the 0th element is the path to the image and 
                                    the 1st element is the label

    - model : the torchvision model from which the data is computed
    """

    # hard coded parameters
    grid_size = 28
    batch_size = 128
    # device = 'cuda'
    img_size = 224

    # transforms
    # standard imagenet normalization
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # Standard imagenet transforms
    preprocessing = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(img_size),
        torchvision.transforms.ToTensor(),
        normalize,
    ])


    # model
    net = torch.nn.DataParallel(model, device_ids=[0, 1])
    net.to(f'cuda:{net.device_ids[0]}')

    # prepare the images
    images = [Image.open(image[0]).convert('RGB') for image in sample_list]# open those that will be proceeded


    # generate the samples and the labels
    x = torch.stack([preprocessing(im) for im in images])
    y = [image[1] for image in sample_list]

    wavelet = WaveletSobol(net, grid_size = grid_size, nb_design=8, batch_size = batch_size, opt = {'approximation' : False, 'size' : grid_size})
    explanations = wavelet(x,y)

    return explanations

def export_explanations(explanations, image_names, target_dir):
    """
    exports the wcams
    """
    # export
    def NormalizeData(data):
        """helper to normalize in [0,1] for the plots"""
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    for explanation, target in zip(explanations, image_names):

        destination = os.path.join(target_dir, target)
        normalized_map = NormalizeData(explanation)
        img = Image.fromarray((normalized_map * 255).astype(np.uint8))
        img.save(destination)

def main():
    """
    main function: computes the wcam 
    """

    # initial list of samples
    initial_list = generate_target_samples(target, count, items)
    completed = os.listdir(target_dir)

    # number of reps
    # reps = int(np.ceil((len(initial_list) - len(completed)) / batch))
    reps = int(count / batch)

    for i in range(reps):

        print('Batch ........................ {}/{}'.format(i+1, reps))

        sample_list, image_names = update_sample_list(initial_list, completed, source_dir, items, batch)
        print('{} WCAMs have been generated, {} remain.'.format(len(completed), len(initial_list) - len(completed)))


        print('Computation of the wcams ...')
        explanations = compute_wcams(sample_list, model)
        print('Computation complete.')
        print('Saving ...')
        export_explanations(explanations, image_names, target_dir)

        # compute the list of samples that are completed
        completed = os.listdir(target_dir)
        print(('Iteration done.'))


if __name__ == '__main__':

    # Run the pipeline.
    main()