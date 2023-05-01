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
from PIL import Image
import os
from utils import helpers
from spectral_sobol.torch_explainer import WaveletSobol
import json


# args
parser = argparse.ArgumentParser(description = 'Computation of the robustness depth')

parser.add_argument('--perturbation', default = 'corruptions', help = "corruption considered", type=str)
parser.add_argument('--sample_size', default = 100, help = "number of images to consider", type=int)
parser.add_argument('--grid_size' , default = 32, type = int)
parser.add_argument('--nb_design', default = 4, type = int)
parser.add_argument('--batch_size', default = 128, type = int)

args = parser.parse_args()

perturbation = args.perturbation
sample_size  = args.sample_size
grid_size = args.grid_size
nb_design = args.nb_design
batch_size = args.batch_size

# parameters

cases = ['vit', 'pixmix', 'adv_free', 'fast_adv', 'adv']

device = 'multi'
imagenet_directory = '../../data/ImageNet/'
target_dir = '../../data/spectral-attribution-outputs'

# load the model

models = []

for case in cases:
    model = helpers.load_model(case, device)
    models.append(model)

# load the images: a sample_size sample from ImageNet val set.
# labels
labels_complete = helpers.format_dataframe(imagenet_directory)

# restrict to a set of labels
np.random.seed(42)
samples = np.random.choice(labels_complete['name'].values, size = sample_size)

# final list of images
labels = labels_complete[labels_complete['name'].isin(samples)]

# load the images
raw_images = [
    Image.open(os.path.join(imagenet_directory, name)).convert('RGB') for name in labels['name'].values
]

# vector of labels to compute the explanations
y = labels['label'].values.astype(np.uint8)

# transforms and transformed images
baseline_transforms = torchvision.transforms.Compose([
torchvision.transforms.Resize(256),
torchvision.transforms.CenterCrop(224)
])


zoomed_in = torchvision.transforms.Compose([
torchvision.transforms.Resize(512),
torchvision.transforms.CenterCrop(224)
])

normalize = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
    ])

images_baseline = [
    baseline_transforms(im) for im in raw_images
]

images_zoomed = [
    zoomed_in(im) for im in raw_images
]

# convert as tensors, ready for computation
x_baseline = torch.stack([
    normalize(im) for im in images_baseline
])

x_zoom = torch.stack([
    normalize(im) for im in images_zoomed
])

# now we compute the explanations for each case

opt = {'size' : grid_size}

results = {}

# add the name of the images considered to simplify the visualization
results['images'] = labels['name'].values.tolist()

for model, case in zip(models, cases):
    print('Case ................ {}'.format(case))

    # define the two wavelets
    wavelet_baseline = WaveletSobol(model, grid_size = grid_size, nb_design = nb_design, batch_size = batch_size, opt = opt, levels = 3)
    wavelet_zoom = WaveletSobol(model, grid_size = grid_size, nb_design = nb_design, batch_size = batch_size, opt = opt, levels = 4)

    expl_baseline = wavelet_baseline(x_baseline, y)
    expl_zoom = wavelet_zoom(x_zoom, y)

    # now we compute the distribution of the importance of the coefficient accross the diffrent levels
    importance_baseline = helpers.level_contributions(expl_baseline, 3)
    importance_zoom = helpers.level_contributions(expl_zoom, 4)

    # add to the results
    results[case] = {
        "regular" : importance_baseline,
        "zoomed"  : importance_zoom
    }

# export the file
with open(os.path.join(target_dir, 'zoom_importance.json'), 'w') as f:
    json.dump(results, f, cls = helpers.NpEncoder)