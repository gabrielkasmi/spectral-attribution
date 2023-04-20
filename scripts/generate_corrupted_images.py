#!/usr/bin/env python
# -*- coding: utf-8 -*-

# generates the corrupted images 
# from the imagenet dataset and the list of images
# for which a WCAM has been computed

# library imports
import sys
sys.path.append('../')

import argparse
from utils import corruptions, make_imagenetc
import os
import tqdm

# args
parser = argparse.ArgumentParser(description = 'Generation of corrupted images')

parser.add_argument('--corruption_dir', default = '../../data/corrupted_images', help = "directory where the samples will be stored", type=str)
parser.add_argument('--backbone', default = 'resnet', help = "name of the backbone under consideration", type=str)
parser.add_argument('--wcams_dir', default = '../../data/wcams-imagenet', help = "directory where the wcams are stored", type=str)
parser.add_argument('--source_dir', default = '../../data/ImageNet', help = "directory where IN validation set is located", type=str)

args = parser.parse_args()

wcams_dir = args.wcams_dir
backbone = args.backbone
source_dir = args.source_dir
target_dir = args.corruption_dir


# dictionnary with all the corruptions considered
corruption_functions = {
    'motion-blur'            : make_imagenetc.motion_blur,
    'defocus-blur'           : make_imagenetc.defocus_blur,
    'brightness'             : make_imagenetc.brightness,
    'spatter'                : make_imagenetc.spatter,
    'jpeg'                   : make_imagenetc.jpeg_compression,
    'saturate'               : make_imagenetc.saturate,
    'pixelate'               : make_imagenetc.pixelate,
    'impulse-noise'          : make_imagenetc.impulse_noise,
    'gaussian-noise'         : make_imagenetc.gaussian_noise,
    'contrast'               : make_imagenetc.contrast,
    'glass-blur'             : make_imagenetc.glass_blur,
    'elastic-transformation' : make_imagenetc.elastic_transform,
    'shot-noise'             : make_imagenetc.shot_noise,
    'gaussian-blur'          : make_imagenetc.gaussian_blur,
}


# if they do not exist, generate the directoris

for dest in corruption_functions.keys():
    if not os.path.exists(os.path.join(target_dir, dest)):
        os.mkdir(os.path.join(target_dir, dest))

# images for which a wcam has been computed
# corruptions will be computed for these images
images_list = os.listdir(os.path.join(wcams_dir, backbone)) 

def main():

    for image in tqdm.tqdm(images_list):

        try : # in some cases it might not work. In this case, we skip to the next image. 
            corrupted_images = corruptions.generate_corruptions(os.path.join(source_dir,image)) 

            # save the images
            for corrupted_image, dest in zip(corrupted_images, corruption_functions.keys()):

                # set the folder name and save the image
                folder_name = os.path.join(target_dir, dest)
                corrupted_image.save(os.path.join(folder_name, image))

        except:
            continue

if __name__ == '__main__':

    # Run the pipeline.
    main()
