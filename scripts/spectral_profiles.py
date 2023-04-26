#!/usr/bin/env python
# -*- coding: utf-8 -*-

# computes the WCAM on the source and a set of perturbed
# images on which the model was fooled bu the perturbation
# first generates an image list

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

# arguments
parser = argparse.ArgumentParser(description = 'Computation of the spectral signatures')

parser.add_argument('--perturbation', default = 'editing', help = "corruption considered", type=str)
parser.add_argument('--sample_size', default = 10, help = "number of images to consider", type=int)
parser.add_argument('--grid_size' , default = 4, type = int)
parser.add_argument('--nb_design', default = 2, type = int)
parser.add_argument('--batch_size', default = 128, type = int)


args = parser.parse_args()


perturbation = args.perturbation
sample_size = args.sample_size
batch_size = args.batch_size
grid_size = args.grid_size
nb_design = args.nb_design

# helper to open the labels

def read_label(path):

    labels_raw = open(os.path.join(path, 'labels.txt')).read().split('\n')
    labels_dict = {
        r.split('\t')[0] : int(r.split('\t')[1]) for r in labels_raw
    }
    labels_true = pd.DataFrame.from_dict(labels_dict, orient = "index").reset_index()
    labels_true.columns = ['name', 'label']

    return labels_true




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

device = 'cuda:2'
# model set up
# model (and case)
# model zoo features : 
#               - RT : 'augmix', 'pixmix', 'sin' (highest accuracy on ImageNet-C), 
#               - AT : 'adv_free, fast_adv and adv,
#               - ST : standard training ('baseline')

models_dir = '../../models/spectral-attribution-baselines'
# cases = ['pixmix', 'baseline', 'augmix', 'sin', 'adv_free', 'fast_adv', 'adv']
cases = ['baseline', 'sin', 'adv']

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
    # frog example : ILSVRC2012_val_00017138.JPEG

elif perturbation == "editing":
    # take samples in the imagenet_e directory
    labels = read_label(imagenet_e_directory)


def main():


    #############################################################################################################
    #                                                                                                           #
    #                                         FIRST PART                                                        #
    #                       Identify and save perturbed images that affect the model's predictions              #
    #                                                                                                           #
    #############################################################################################################

    # transforms
    resize_and_crop = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224)
    ])

    # first go into the folder where the images are stored
    target_folder = os.path.join(target_dir, perturbation)
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    # get the list of images that have been created
    img_names = os.listdir(target_folder)

    # generate the transformed images only if we still miss some
    # images

    if len(img_names) < sample_size:

        # transforms
        normalize = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
        ])

        images_list = labels['name'].values

        n_samples = 4 * batch_size
        completed_samples = len(img_names)

        # number of images that have been created so far
        count = completed_samples
        completed_images = [im for im in img_names]
        
        # loop until the desired number of images is completed
        while count < sample_size:

            # remaining items 
            remaining_items = [i for i in images_list if not i in completed_images]
            print(len(remaining_items))

            # initialize the target names
            np.random.seed(42)
            target_names = np.random.choice(remaining_items, 2)

            # source images
            source_images = [Image.open(os.path.join(imagenet_directory, img_name)).convert('RGB') for img_name in target_names]
            source_images = [resize_and_crop(im) for im in source_images]

            # perturbed images
            # compute the set of perturbed images
            if perturbation == "corruptions":
                perturbed_images = {
                    img_name : helpers.compute_corrupted_images(image) for img_name, image in zip(target_names, source_images)
                }

            elif perturbation == "editing":
                perturbed_images = {
                    img_name : helpers.retrieve_edited_samples(imagenet_e_directory, img_name) for img_name in target_names
                }

            else:
                raise ValueError

            print('Corrupted images generated, now evaluating')
            # x_source
            x_source = torch.stack([
                normalize(im) for im in source_images
            ])

            # evaluate the models on the samples
            # results will store the predictions on the perturbed images       
            results = {
                case : {} for case in cases
            }


            for case, model in zip(cases, models):

                # source predictions
                source_preds = helpers.evaluate_model_on_samples(x_source, model, batch_size)

                # predictions on the altered samples
                target_preds = {}

                for image in perturbed_images.keys():
                    
                    images = perturbed_images[image]
                    x_pert = torch.stack([
                        normalize(im) for im in images
                    ])

                    target_preds[image] = helpers.evaluate_model_on_samples(x_pert, model, batch_size)

                # now that the inference is complete, for each image we record the 
                # predictions that have changed

                changed_predictions = {}

                for i, img_name in enumerate(list(target_preds.keys())):

                    # source prediction
                    source_y = source_preds[i]
                    target_y = target_preds[image]

                    # identify the indices for which the prediction has changed
                    changed_indices = np.where(target_y != source_y * np.ones(len(target_preds)))[0]

                    # save the changed indices for the current image
                    changed_predictions[img_name] = changed_indices

                # add the changed predictions to the dictionnary of results
                results[case] = changed_predictions
            # print('Results dictionnary')
            # print(results)

            # now that we have the effect of the perturbation across models
            # we consider the intersection of all indices list
            # print('Evaluation complete. Now keeping the images')

            images_to_keep = {}

            for i, image_name in enumerate(target_names):

                preds = []
                for case in cases:
                    temp_preds = results[case][image_name]
                    preds.append(set(temp_preds))

                # take the intersection 
                intersection = helpers.return_intersection(preds)

                if len(intersection) == 0: # no intersection between the index, continue
                    continue
                else:
                    # add the images 
                    images_to_keep[image_name] = [
                        source_images[i]
                    ]
                    # add the pertutbred images to be found at these indices
                    for index in intersection:
                        images_to_keep[image_name].append(perturbed_images[image_name][index])

            print('Results dataframe to keep')
            print(results)

            # save the images to keep
            if len(images_to_keep.keys()) > 0:
                for image_name in images_to_keep.keys():

                    # create the directory
                    dest = os.path.join(target_folder, image_name)
                    os.mkdir(dest)
                    # save the source

                    images = images_to_keep[image_name]
                    images[0].save(os.path.join(dest, "source.png"))
                    
                    # save the altered images 
                    for i, image in enumerate(images[1:]):
                        image.save(os.path.join(dest, 'altered_{}.png'.format(i)))


                # update the list of images that have been proceeded
            for t in target_names:
                completed_images.append(t)

            # update the number of images created
            count += len(images_to_keep.keys())
            print('{} perturbed samples created. {} remain.'.format(count, sample_size - count))

    print('Database instantiation complete. Now computing the WCAMs.')

    #############################################################################################################

    #############################################################################################################
    #                                                                                                           #
    #                                        SECOND PART                                                        #
    #                       Open the images and compute the spectral CAM                                        #
    #                                                                                                           #
    #############################################################################################################


    # go into the folder and open the images
    # update the list of images
    img_names = os.listdir(target_folder)

    # transforms
    normalize = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])

    for image_name in img_names:

        dest = os.path.join(target_folder, image_name)

        print('Image .................... {}'.format(image_name))

        for case, model in zip(cases, models):

            print('Case ................... {}'.format(case))

            images_data = [t for t in os.listdir(os.path.join(target_folder, image_name)) if t[-4:] == '.png']
            images = [Image.open(os.path.join(dest,im)).convert('RGB') for im in images_data]

            x = torch.stack([
                normalize(im) for im in images
            ])

            y = helpers.evaluate_model_on_samples(x, model, batch_size).astype(np.uint8)

            # Compute the explanations
            wavelet = WaveletSobol(model, grid_size = grid_size, nb_design = nb_design, batch_size = batch_size)
            explanations = wavelet(x,y)
            spatial_cams = wavelet.spatial_cam

            # export them 
            destination = os.path.join(os.path.join(os.path.join(target_folder, image_name)), case)
            if not os.path.exists(destination):
                os.mkdir(destination)

            # save the images in the folder
            helpers.postprocess(explanations, spatial_cams, destination, images_data)


if __name__ == '__main__':
    main()