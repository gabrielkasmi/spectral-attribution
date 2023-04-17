# -*- coding: utf-8 -*-

# libraries
from .make_imagenetc import *
import numpy as np
from PIL import Image


np.random.seed(42)

# dictionnary of corruptions to consider
corruption_functions = {
    'motion-blur'            : motion_blur,
    'defocus-blur'           : defocus_blur,
    'brightness'             : brightness,
    'spatter'                : spatter,
    #'zoom-blur'              : make_imagenetc.zoom_blur,
    'jpeg'                   : jpeg_compression,
    'saturate'               : saturate,
    'pixelate'               : pixelate,
    'impulse-noise'          : impulse_noise,
    #'frost'                  : make_imagenetc.frost,
    'gaussian-noise'         : gaussian_noise,
    'contrast'               : contrast,
    'glass-blur'             : glass_blur,
    'elastic-transformation' : elastic_transform,
    'shot-noise'             : shot_noise,
    'gaussian-blur'          : gaussian_blur,
    # 'snow'                   : make_imagenetc.snow,
    #"fog"                    : make_imagenetc.fog
}

corruptions = len(corruption_functions.keys())

def generate_corruptions(path):
    """
    generates a corruption in the ImageNet-C style for the image path passed as input

    returns a list with all corruptions and the original image at index 0
    """

    img = Image.open(path).convert('RGB')

    images = [img] # list returned, will store the images and the corruptions 

    dists = {
            c : corruption_functions[c](img, severity = k) for (c, k) in zip(corruption_functions.keys(), np.random.randint(1,6, size = corruptions))
        }
    
    # get the images in the correct format
    for c in dists.keys():
        
        img = dists[c]
        
        if isinstance(img, np.ndarray):
            d = img.astype(np.uint8)
            dists[c] = Image.fromarray(d).convert('RGB')


    for c in dists.keys():
        images.append(dists[c])

    return images