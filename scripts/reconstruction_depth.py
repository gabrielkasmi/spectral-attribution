#!/usr/bin/env python
# -*- coding: utf-8 -*-

# computes the reconstruction depth on source and 
# altered samples over a sampled set of imagenet instances
# if the perturbation is corruption, computes the perturbation on the fly
# if the perturbation is editing, retrieves the list of edited images beforehand

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
import json


# args
parser = argparse.ArgumentParser(description = 'Computation of the robustness depth')

parser.add_argument('--perturbation', default = 'corruptions', help = "corruption considered", type=str)
parser.add_argument('--sample_size', default = 10, help = "number of images to consider", type=int)
parser.add_argument('--grid_size' , default = 8, type = int)
parser.add_argument('--nb_design', default = 2, type = int)
parser.add_argument('--batch_size', default = 128, type = int)

args = parser.parse_args()

perturbation = args.perturbation
sample_size  = args.sample_size
grid_size = args.grid_size
nb_design = args.nb_design
batch_size = args.batch_size

# directories
# parameters and directories
imagenet_e_directory = '../../data/ImageNet-E/'
imagenet_directory = '../../data/ImageNet/'

target_dir = '../../data/spectral-attribution-outputs'

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
cases = ['pixmix', 'baseline', 'augmix']#, 'sin', 'adv_free', 'fast_adv', 'adv']

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

# helper to open the labels

def read_label(path):

    labels_raw = open(os.path.join(path, 'labels.txt')).read().split('\n')
    labels_dict = {
        r.split('\t')[0] : int(r.split('\t')[1]) for r in labels_raw
    }
    labels_true = pd.DataFrame.from_dict(labels_dict, orient = "index").reset_index()
    labels_true.columns = ['name', 'label']

    return labels_true

# list of images

# images
if perturbation == "corruptions":
    # labels dataframe
    labels = helpers.format_dataframe(imagenet_directory)

    # generate the images
    np.random.seed(1)
    img_names = np.random.choice(labels['name'].values, sample_size)

    # restrict the labels
    labels_true = labels[labels['name'].isin(img_names)]

elif perturbation == "editing":
    # take samples in the imagenet_e directory
    labels = read_label(imagenet_e_directory)

    # generate the images
    np.random.seed(1)
    img_names = np.random.choice(labels['name'].values, sample_size)
    labels_true = labels[labels['name'].isin(img_names)]




def main():

    opt = {'size' : grid_size} # dont upsample the wcam

    for model, case in zip(models, cases): 

        print('Case ............... {}'.format(case))

        # initilize the explainer
        wavelet = WaveletSobol(model, grid_size = grid_size, nb_design = nb_design, batch_size = batch_size, opt = opt)

        # compute the robustness
        robustness = helpers.compute_robustness_and_depth(img_names, imagenet_directory, imagenet_e_directory, target_dir, labels_true, wavelet, model, batch_size = batch_size, perturbation = perturbation)

        # save it as a dictionnary in the target dir
        filename = 'reconstruction_depth_{}_{}.json'.format(case, perturbation)
        with open(os.path.join(target_dir, filename), 'w') as f:
            json.dump(robustness, f, cls = helpers.NpEncoder)

if __name__ == '__main__':

    # Run the pipeline.
    main()