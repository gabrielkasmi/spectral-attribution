# -*- coding: utf-8 -*-

# libraries
import pandas as pd
import numpy as np
from PIL import Image
import os
import pywt
import cv2
import torchvision
from .corruptions import *


np.random.seed(42)

def format_dataframe(directory, filter = None):
    """
    helper that converts the txt file into a dataframe
    if filter is not none, returns the dataframe with only entries
    comprised in the list filter

    returns a pd.Dataframe
        """
    labels_raw = open(os.path.join(directory, 'val.txt')).read().split('\n')
    labels_dict = {r[:28] : int(r[29:]) for r in labels_raw if r[:10] == 'ILSVRC2012'}
    labels_true = pd.DataFrame.from_dict(labels_dict, orient = "index").reset_index()
    labels_true.columns = ['name', 'label']
    
    if filter is not None:
        labels_true = labels_true[labels_true['name'].isin(filter)] # restrict to the images studied
    
    return labels_true



def compute_robustness_and_depth(source_img_names, imagenet_dir, labels_true, wavelet, model, batch_size = 128):
    """
    wrapper that computes for a set of images the robustness and the reconstruction depth, in two cases

    computation of the robustness:
    * share of correct predictions over a set of corrupted instances

    computation of the reconstruction depth:
    * reconstruction from sobol indices by decreasing order. Corresponds to the first image that is correctly
      predicted by the model
    * baseline image: either the true image or a corrupted image. In this case, the initial prediction is taken as the 
      'true label' to recover from the reconstruction

      
    args:
    - source_img_names: the name of the source images
    - imagenet_dir : the directory where the images are stored
    - labels_true : a pd.DataFrame of labels
    - wavelet : the WaveletSobol explainer

    returns : 
    - robustness : a list with the robustnesses (comprised between 0 and 1)
    - reconstruction depth: a dictoinnary with the reconstruction depths (integers) 
                            keys corresponds to the cases ('source') or 'corrupted'

    """

    # misc transforms
    resize_and_crop = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224)
    ])


    # transforms
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    preprocessing = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize,
    ])

    # source images
    source_images = [Image.open(os.path.join(imagenet_dir, img_name)).convert('RGB') for img_name in source_img_names]
    source_images = [resize_and_crop(im) for im in source_images]


    # perturbed images
    # compute the set of perturbed images
    perturbed_images = {
        img_name : compute_corrupted_images(image) for img_name, image in zip(source_img_names, source_images)
    }

    # compute the robustness and the reconstruction depth
    robustness = []
    reconstruction_depth = {
        'source'  : [],
        'corrupted' : []
    }

    for i, name in enumerate(list(perturbed_images.keys())):

        # retrive the images
        images = perturbed_images[name]

        # transforms and convert as a tensor
        x = torch.stack(
            [preprocessing(img) for img in images]
        )

        # label
        label = labels_true[labels_true['name'] == name]['label'].values

        # evaluate the model and compute the robustness 
        # append it to the list
        preds = evaluate_model_on_samples(x, model, batch_size)
        correct = sum(preds == label * np.ones(len(preds)))
        robustness.append(correct / len(preds))

        # compute the reconstruction depth on the source image and a random perturbed image
        source_image = source_images[i]
        index = np.random.randint(1,len(images), size = 1).item()
        perturbed_image = images[index]

        # compute the wcam
        imgs = torch.stack(
            [preprocessing(im) for im in [source_image, perturbed_image]]
        )

        # label
        y = (label * np.ones(2)).astype(np.uint8)

        # compute the explanations
        explanations = wavelet(imgs, y)

        # altered images: images that are reconstructed from the sobol index
        reconstructed_images = {
            'source' : reconstruct_images(np.array(source_image), explanations[0]),
            'corrupted' : reconstruct_images(np.array(perturbed_image), explanations[1])
        }

        for case in reconstructed_images.keys():

            images = reconstructed_images[case]

            x = torch.stack(
                [preprocessing(img) for img in images]  
            )

            reconstructed_preds = evaluate_model_on_samples(x, model, batch_size)

            if case == 'source':
            # we consider the true label as the baseline

                # find the index of the first correct prediction
                try:
                    depth = np.min(np.where(reconstructed_preds == label * np.ones(len(reconstructed_preds)))[0])
                except ValueError:
                    depth = np.nan
            else:
                # we consider the prediction corresponding to the 
                # altered image (in the baseline case) as the 'true label'
                est_label = preds[index]
                    # find the index of the first correct prediction
                try:
                    depth = np.min(np.where(reconstructed_preds == est_label * np.ones(len(reconstructed_preds)))[0])
                except ValueError:
                    depth = np.nan

            reconstruction_depth[case].append(depth)

    return robustness, reconstruction_depth





def compute_corrupted_images(img):
    """
    computes a list of images altered following the ImageNet-C corruption procedure
    we loop through all corruptions and all strenghts. 
    """

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

    corruptions = corruption_functions.keys()

    dists = {}
    for c in corruptions:
        dists[c] = []
        for i in range(6):
            # compute the perturbation
            pert = corruption_functions[c](img, severity = i)
            # convert as PIL image if necessary
            if isinstance(pert, np.ndarray):
                d = pert.astype(np.uint8)
                pert = Image.fromarray(d).convert('RGB')

            # add to the lsit
            dists[c].append(pert)
    
    # flatten the list
    # the 6 first images correspond to the first perturbation, then six second to the second eprturbation
    # etc.
    return list(sum([dists[c] for c in dists.keys()], []))



def NormalizeData(data):
    """helper to normalize in [0,1] for the plots"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def perturb_and_invert(mask, slices, transform, wavelet = "haar"):
    """
    computes the perturbed wavelet transform and inverts it to return the 
    transformed image
    
    the mask's size whould match the size of the transform.
    
    returns the rgb image (as an array)
    """    
    perturbed_image = np.zeros(transform.shape)
        
    for i in range(perturbed_image.shape[2]):
        
        # apply a channel wise perturbation
        perturbation = transform[:,:,i] * mask
        
        # using the slices passed as input and the perturbed
        # transform, compute the inverse for this channel
        
        # compute the coeffs
        coeffs = pywt.array_to_coeffs(perturbation, slices, output_format = "wavedec2")
        perturbed_image[:,:,i] = pywt.waverec2(coeffs, wavelet)
        
    return (NormalizeData(perturbed_image) * 255).astype(np.uint8)

def inverse_wavelet(wavelet_transform, mask, wavelet_type = "haar"):
    # compute the wavelet transform
    out, slices = wavelet_transform

    #reshape the masks to the size of the image
    # mask = cv2.resize(mask, out.shape[:2], interpolation = cv2.INTER_NEAREST)

    # compute the perturbation and append to the list
    pert = perturb_and_invert(mask, slices, out, wavelet = wavelet_type)


    return Image.fromarray(pert)

def compute_wavelet_transform(image, level = 3, wavelet = 'haar'):
    """
    computes the wavelet transform of the input image
    
    returns a (W,H,C) array where for each channel we compute 
    the associated wavelet transform
    returns the slices as well, to facilitate reconstruction
    
    remark: better to stick with haar wavelets as they do not induce 
    shape issues between arrays.
    """
    
    if not isinstance(image, np.ndarray):
        
        image = np.array(image)
        
    
    transform = np.zeros(image.shape)
    
    for i in range(image.shape[2]):
        # compute the transform for each channel
        
        x = image[:,:,i]

        coeffs = pywt.wavedec2(x, wavelet, level=level)
        arr, slices = pywt.coeffs_to_array(coeffs)
        
        transform[:,:,i] = arr
        
    return transform, slices

def compute_sparse_masks(arr):
    """
    returns a sequence of binary masks with increasing coefficients
    """

    # small routine to convert the mask as a uint8 image
    if arr.dtype == 'float32':
        normalized_map = NormalizeData(arr)
        arr = (normalized_map * 255).astype(np.uint8)

    # Get the flattened indices of the elements in descending order
    flat_indices = np.argsort(arr.ravel())[::-1]

    # Convert the flattened indices to 2D coordinates
    x, y = np.unravel_index(flat_indices, arr.shape)
    
    # number of positive coordinates
    n_pos = len(np.where(arr > 0)[0])

    # masks: for each additional coordinate 
    n_masks = n_pos + 1

    # set the sequence of masks
    masks = np.zeros((arr.shape[0], arr.shape[1], n_masks)).astype(np.uint8)

    for i in range(n_masks):

        if i+1 < n_masks: # add the coordinates one by one
            masks[x[:(i +1)], y[:(i +1)], i] = 255

        else:
            masks[:,:,i] = 255

    return masks

def reconstruct_images(image, cam, levels = 3):
    """
    reconstructs the image  passed as input
    py picking the importance coefficietns 
    by decreasing order
    
    returns n_pos + 1 images, the last one 
    being reconstructed with all coefficients 
    """

    # compute the wt of the image
    wt = compute_wavelet_transform(image, level = levels)

    input_shape = image.shape[:2]


    # compute the masks
    masks = compute_sparse_masks(cam)

    images = []

    for i in range(masks.shape[-1]):

        # upsample the mask
        mask = masks[:,:,i]
        upsampled_mask = cv2.resize(mask, input_shape, interpolation = cv2.INTER_NEAREST)

        # invert the image
        pert = inverse_wavelet(wt, upsampled_mask)

        # add to the list
        images.append(pert)

    return images

# set of auxiliary functions
def plot_wavelet_regions(size,levels):
    """
    returns the dictonnaries with the
    coordinates of the lines for the plots
    """

    center = size // 2
    h, v = {}, {} # dictionnaries that will store the lines
    # initialize the first level
    h[0] = np.array([
        [0, center],
        [size,center],
    ])
    v[0] = np.array([
        [center,size],
        [center,0],
    ])
    # define the horizontal and vertical lines at each level
    for i in range(1, levels):
        h[i] = h[i-1] // 2
        h[i][:,1]
        v[i] = v[i-1] // 2
        v[i][:,1] 
        
    return h, v   


def compute_average_classes(labels_wcam, wcams_dir, grid_size = 28):
    """
    averages the WCAMs computed for different classes in the 
    imagenet validation set.

    args
    labels_wcam (pd.DataFrame): the dataframe that contains the labels of the samples
    wcams_dir (str) : the directory where the wcams are located
    grid_size (int, optional) : the size of the grid. Corresponds to the parameter passed as input
                                of the WaveletSobol object.

    returns : 
    - classes (dict) : contains the averaged WCAM for each class 
    """

    labels_list = np.unique(labels_wcam['label'].values) # labels contained in the sample

    # compute the averages

    # output dictionnary:
    classes = {}

    for label in labels_list:

        # retrieve the corresponding images:
        images = labels_wcam[labels_wcam['label'] == label]['name'].values

        classes[label] = np.zeros((grid_size, grid_size))

        # add the class activation maps
        for image in images:
            img = np.array(Image.open(
                os.path.join(wcams_dir, image)).convert('L')
            ).astype(float)

            classes[label] += img / 255 

        # divide by the number of images for that label to get the mean
        classes[label] /= len(images)

    return classes

def add_lines(size, levels, ax):
    """
    add white lines to the ax where the 
    WCAM is plotted
    """
    h, v = plot_wavelet_regions(size, levels)

    ax.set_xlim(0,size)
    ax.set_ylim(size,0)

    for k in range(levels):
        ax.plot(h[k][:,0], h[k][:,1], c = 'w')
        ax.plot(v[k][:,0], v[k][:,1], c = 'w')

    return None


def evaluate_model_on_samples(x, model, batch_size):
    """
    inference loop of a model on a set of samples whose size
    can exceed the batch size.

    returns the vector of predicted classes
    """

    # retrieve the device
    device = next(model.parameters()).device
    
    # predictions vector
    y = np.empty(len(x))

    # nb batch
    nb_batch = int(np.ceil(len(x) / batch_size))

    model.eval()

    with torch.no_grad():

        for batch_index in range(nb_batch):
            # retrieve masks of the current batch
            start_index = batch_index * batch_size
            end_index = min(len(x), (batch_index+1)*batch_size)

            # batch samples
            batch_x = x[start_index : end_index, :,:,:].to(device)

            # predictions
            preds = model(batch_x).cpu().detach().numpy()
            batch_y = np.argmax(preds, axis = 1)

            # store the results
            y[start_index:end_index] = batch_y

    return y