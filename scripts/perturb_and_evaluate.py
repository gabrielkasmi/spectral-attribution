#!/usr/bin/env python
# -*- coding: utf-8 -*-

# perturbs a set of images according to a given
# perturbation and computes the wcam of the source image and the affected images
# stores everything in a dedicated directory passed as input by the user

# library imports
import sys
sys.path.append('../')

import argparse
import numpy as np
import torchvision
import torch
from torchvision.models import resnet50
from PIL import Image
import os
from utils import helpers
from spectral_sobol.torch_explainer import WaveletSobol
import pandas as pd


# helper to open the labels

def read_label(path):

    labels_raw = open(os.path.join(path, 'labels.txt')).read().split('\n')
    labels_dict = {
        r.split('\t')[0] : int(r.split('\t')[1]) for r in labels_raw
    }
    labels_true = pd.DataFrame.from_dict(labels_dict, orient = "index").reset_index()
    labels_true.columns = ['name', 'label']

    return labels_true


# arguments
parser = argparse.ArgumentParser(description = 'Computation of the spectral signatures')

parser.add_argument('--perturbation', default = 'editing', help = "corruption considered", type=str)
parser.add_argument('--sample_size', default = 100, help = "number of images to consider", type=int)
parser.add_argument('--grid_size' , default = 28, type = int)
parser.add_argument('--nb_design', default = 8, type = int)
parser.add_argument('--batch_size', default = 128, type = int)


args = parser.parse_args()

##########################################################################
#                                                                        #
#                            MODEL SET UP                                #
#                                                                        #
##########################################################################

# CHANGE HERE TO SWICTH MODEL BACKBONE

device = 'cuda:1'
# model set up
# model (and case)
# model zoo features : 
#               - RT : 'augmix', 'pixmix', 'sin' (highest accuracy on ImageNet-C), 
#               - AT : 'adv_free, fast_adv and adv,
#               - ST : standard training ('baseline')

models_dir = '../../models/spectral-attribution-baselines'
cases = ['pixmix', 'baseline', 'augmix', 'sin', 'adv_free', 'fast_adv', 'adv']

# load the models
models = []
for case in cases:
    
    if case == 'baseline':
        model = resnet50(pretrained = True).to(device).eval()
    elif case in ['augmix', 'pixmix', 'sin']:
        model = resnet50(pretrained = False) # model backbone #torch.load(os.path.join(models_dir, '{}.pth'.format(case))).eval()
        weights = torch.load(os.path.join(models_dir, '{}.pth.tar').format(case))
        model.load_state_dict(weights['state_dict'], strict = False)
        model.eval()
    elif case in ['adv_free', 'fast_adv', 'adv']:
        model = resnet50(pretrained = False) # model backbone #torch.load(os.path.join(models_dir, '{}.pth'.format(case))).eval()
        weights = torch.load(os.path.join(models_dir, "{}.pth".format(case)))
        model.load_state_dict(weights)
        model.eval()

    # we will loop over the models
    models.append(model)

##########################################################################


# directories
# parameters and directories
imagenet_e_directory = '../../data/ImageNet-E/'
imagenet_directory = '../../data/ImageNet/'

target_dir = '../../data/spectral-attribution-outputs'


# parameters
sample_size  = args.sample_size
perturbation = args.perturbation
grid_size    = args.grid_size
nb_design    = args.nb_design
batch_size   = args.batch_size

# update the target_dir
root_target = os.path.join(target_dir, perturbation)

# create this directory
if not os.path.exists(root_target):
    os.mkdir(root_target)


# transforms
normalize = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224)
])


# images
if perturbation == "corruptions":
    # labels dataframe
    labels = helpers.format_dataframe(imagenet_directory)

    # generate the images
    np.random.seed(1)
    img_names = np.random.choice(labels['name'].values, sample_size)
    samples = [
        preprocess(Image.open(os.path.join(imagenet_directory, img_name)).convert('RGB')) for img_name in img_names
    ]

elif perturbation == "editing":
    # take samples in the imagenet_e directory
    labels = read_label(imagenet_e_directory)

    # generate the images
    np.random.seed(42)
    img_names = np.random.choice(labels['name'].values, sample_size)
    samples = [
        preprocess(Image.open(os.path.join(imagenet_directory, img_name)).convert('RGB')) for img_name in img_names
    ]

def main():

    # options dictionnaries for the explainer
    # and the main function
    opt = {
        'batch_size' : batch_size,
        'imagenet_e_directory' : imagenet_e_directory,
        'normalization' : normalize
    }

    params = {
        'size' : grid_size
    }


    for model, case in zip(models, cases):

        print('Case ........... {}'.format(case))

        wavelet = WaveletSobol(model, grid_size = grid_size, nb_design = nb_design, batch_size = batch_size, opt = params)

        # evaluate the models and compute the wcams
        wcams, preds = helpers.compute_wcam_on_source_and_altered_samples(model, img_names, samples, wavelet, perturbation, opt)

        # save it
        # create the directory corresponding to the model
        output_dir = os.path.join(root_target, case)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # save 
        helpers.postprocess(wcams, preds, output_dir)
        
if __name__ == '__main__':

    # Run the pipeline.
    main()