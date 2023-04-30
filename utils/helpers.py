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
from scipy.stats import wasserstein_distance
import json
import matplotlib.pyplot as plt
import torch
from spectral_sobol.torch_explainer import WaveletSobol
from torchvision.models import resnet50, vgg16, vit_b_16


def load_model(model_name, device, models_dir = '../../models/spectral-attribution-baselines'):
    """
    loads a model
    """

    if model_name == "vit":
        model = vit_b_16(pretrained = True).to(device).eval()
    elif model_name == "vgg": 
        model = vgg16(pretrained = True).to(device).eval()   
    elif model_name == 'baseline':
        model = resnet50(pretrained = True).to(device).eval()
    elif model_name in ['augmix', 'pixmix', 'sin']:
        model = resnet50(pretrained = False) # model backbone #torch.load(os.path.join(models_dir, '{}.pth'.format(case))).eval()
        weights = torch.load(os.path.join(models_dir, '{}.pth.tar').format(model_name))
        model.load_state_dict(weights['state_dict'], strict = False)
        model.to(device)
        model.eval()
    elif model_name in ['adv_free', 'fast_adv', 'adv']:
        model = resnet50(pretrained = False) # model backbone #torch.load(os.path.join(models_dir, '{}.pth'.format(case))).eval()
        weights = torch.load(os.path.join(models_dir, "{}.pth".format(model_name)))
        model.load_state_dict(weights)
        model.to(device)
        model.eval()

    return model


def find_stable_predictions(image_name, model, directory, perturbation, batch_size = 128):
    """
    perturbs the image passed as input and evaluates the model
    on the perturbed samples. 

    then returns the number of desired images (default : 1) for which 
    the prediction is the same
    """  

    # transforms
    # for the editing case
    resize_and_crop = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224)
    ])

    # normalisation for inference
    normalize = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(os.path.join(directory, 'source.png')).convert('RGB')

    # perturb the image
    # perturbed images
    # compute the set of perturbed images
    if perturbation == "corruptions":
        perturbed_images = compute_corrupted_images(image) 

    elif perturbation == "editing":
        
        perturbed_images = retrieve_edited_samples(directory, image_name)
        perturbed_images = [resize_and_crop(im) for im in perturbed_images]

    # evaluate the model on the perturbed set of images
    # add the unaffected image at the beginning
    perturbed_images.insert(0, image)

    # normalization
    x = torch.stack([
        normalize(im) for im in perturbed_images
    ])

    preds = evaluate_model_on_samples(x, model, batch_size)
    initial_pred = preds[0]

    correct_cases = np.where(np.array(preds[1:]) == initial_pred * np.ones(len(preds) - 1))[0].tolist()

    # retrieve the indices for which the model was not affected
    index = np.random.choice(correct_cases, 1).item()

    return [image, perturbed_images[index]], initial_pred

def plot_wcams_horizontal(images, explanations, labels, case, save = None):
    """
    plots a set of wcams
    """

    levels = 3
    size = 224

    n_cols = len(images)

    fig, ax = plt.subplots(2, n_cols, figsize = (4 * n_cols, 8))
    plt.rcParams.update({'font.size': 17})


    # first row: images
    # second row: wcams
    for i in range(n_cols):
        ax[0,i].imshow(images[i])
        ax[0,i].axis('off')
        title = 'Image \n {}'
        ax[0,i].set_title(title.format(labels[i]))

        ax[1,i].imshow(explanations[i], cmap = 'jet')
        ax[1,i].set_title("WCAM")
        ax[1,i].axis('off')
        add_lines(size, levels, ax[1,i])

    plt.suptitle('Case : {}'.format(case))
    fig.tight_layout()

    if save is not None:

        plt.savefig(save)
        plt.show()
        plt.close()
    
    else:
        plt.show()

    return None

def plot_stable_and_unstable_prediction(image_name, model, directory, corruption, case, labels, save = None):
    """
    finds the wcam on altered samples and coputes the wcam on an unaltered sample

    'case' (str) the case to consider (e.g., baseline, etc)
    'label' (list) labels to be displayed on the plot
    """

    batch_size = 128 if not 'case' == 'vit' else 64

    # get the wcam altered
    indices = [int(t.split('_')[1][:-4]) for t in os.listdir(directory) if t[:7] == 'altered']
    index = np.random.choice(indices, 1).item()
    rank = indices.index(index) + 1

    target = Image.open(os.path.join(directory, "altered_{}.png".format(index)))

    # open the wcams for the cases passed as input

    destination = os.path.join(directory, case)
    target_wcam = Image.open(os.path.join(destination, "wcam_target_{}.png".format(rank))).convert('L')


    # get the stable predictions
    images, pred = find_stable_predictions(image_name, model, directory, corruption)

    # compute the wcam 
    # normalisation for inference
    normalize = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
    ])


    x = torch.stack([normalize(im) for im in images])

    wavelet = WaveletSobol(model, grid_size = 28, nb_design = 8, batch_size = batch_size, opt = {'approximation' : False})
    explanations = wavelet(x, np.array([pred, pred]).astype(np.uint8))

    # complete list of images
    images.append(target)
    explanations.append(np.array(target_wcam))

    # plot everything

    plot_wcams_horizontal(images, explanations, labels, case, save = save)


def plot_wcams(directory, cases, opt = None, save = None):
    """
    plots the wcam of the image image_name for the cases 
    passed as input in the cases inputed in the list

    opt is a set of optional parameters if one wants to specify 
    precise indices 
    """

    # open the source and target images
    source = Image.open(os.path.join(directory, "source.png")).convert('RGB')

    if opt is not None and "index" in opt.keys():
        index = opt['index']

    else: # TODO. Consider a random index in the set (altered_1, ... altered_n)
        indices = [int(t.split('_')[1][:-4]) for t in os.listdir(directory) if t[:7] == 'altered']
        index = np.random.choice(indices, 1).item()
        rank = indices.index(index) + 1

    target = Image.open(os.path.join(directory, "altered_{}.png".format(index)))

    # open the wcams for the cases passed as input

    if len(cases) == 1: # only one case to consider, so 2x2 plot

        destination = os.path.join(directory, cases[0])

        # open the source and target
        # 

        source_wcam = Image.open(os.path.join(destination, "wcam_source.png")).convert('L')
        target_wcam = Image.open(os.path.join(destination, "wcam_target_{}.png".format(rank))).convert('L')

        generate_plot([source, target], [source_wcam], [target_wcam], cases, save = save)
    
    else:
        
        source_wcams = []
        target_wcams = []

        for case in cases:

            destination = os.path.join(directory, case)

            source_wcam = Image.open(os.path.join(destination, "wcam_source.png")).convert('L')
            target_wcam = Image.open(os.path.join(destination, "wcam_target_{}.png".format(rank))).convert('L')

            source_wcams.append(source_wcam)
            target_wcams.append(target_wcam)

        generate_plot([source, target], source_wcams, target_wcams, cases, save = save)

    return None


def generate_plot(images, source_wcams, target_wcams, cases, save = None):
    """
    generates a plot with the images and the wcams
    """

    levels = 3
    size = 224

    n_cols = 1 + len(source_wcams)

    fig, ax = plt.subplots(2, n_cols, figsize = (4 * n_cols, 8))
    # add options here
    plt.rcParams.update({'font.size': 15})

    # first column : images
    source, target = images
    ax[0,0].imshow(source)
    ax[0,0].axis('off')
    ax[0,0].set_title("Source image")

    ax[1,0].imshow(target)
    ax[1,0].axis('off')
    ax[1,0].set_title("Altered image")

    # subsequent columns: wcams
    for i, (source_wcam, target_wcam) in enumerate(zip(source_wcams, target_wcams)):

        title = '{} case \n WCAM on the source image'.format(cases[i])

        ax[0, i + 1].imshow(source_wcam, cmap = 'jet')
        ax[0, i + 1].axis('off')
        ax[0, i + 1].set_title(title)
        add_lines(size, levels, ax[0, i+1])


        ax[1, i + 1].imshow(target_wcam, cmap = 'jet')
        ax[1, i + 1].axis('off')
        ax[1, i + 1].set_title('WCAM on the altered image')
        add_lines(size, levels, ax[1, i+1])

    fig.tight_layout()
    
    if save is not None:

        plt.savefig(save)
        plt.show()
        plt.close()
    
    else:
        plt.show()
    
    return None


def return_intersection(preds):
    """
    returns a loose intersection between the indices
    especially if there are more than three sets, we return 
    the intersection of the non empty sets only
    """

    if len(preds) == 1:
        inter_preds = preds[0]
    elif len(preds) == 2:
        inter_preds = preds[0].intersection(preds[1])

    else:
        inter_preds = preds[0].intersection(preds[1])

        additional_components = []
        for i in range(2, len(preds)):

            additional_components.append(list(preds[i]))

        # get the unique components only and flatten the list
        additional_components = list(set(list(sum(additional_components, []))))

        # append the additional components
        inter_preds = list(inter_preds)
        inter_preds.append(additional_components)

        inter_preds = list(set(list(sum(inter_preds, []))))

    return inter_preds

def initialize_samples(source_img_names, imagenet_dir, imagenet_e_directory, target_dir, perturbation = None):
    """
    initializes the perturbed samples

    returns a dictionnary of perturbed samples
    """

    # misc transforms
    resize_and_crop = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224)
    ])

    # directory set up
    # see if the directory of perturbed images already exists
    directory_name = 'images_{}'.format(perturbation)

    # see if we need to create the images or not
    read = os.path.exists(os.path.join(target_dir, directory_name))

    if not read:
        os.mkdir(os.path.join(target_dir, directory_name))

        # source images
        source_images = [Image.open(os.path.join(imagenet_dir, img_name)).convert('RGB') for img_name in source_img_names]
        source_images = [resize_and_crop(im) for im in source_images]

        # perturbed images
        # compute the set of perturbed images
        if perturbation == "corruptions":
            perturbed_images = {
                img_name : compute_corrupted_images(image) for img_name, image in zip(source_img_names, source_images)
            }

        elif perturbation == "editing":
            perturbed_images = {
                img_name : retrieve_edited_samples(imagenet_e_directory, img_name) for img_name in source_img_names for img_name in source_img_names
            }

        else:
            raise ValueError
        

        # save the images
        for img_name in perturbed_images.keys():

            tmp = os.path.join(os.path.join(os.path.join(target_dir, directory_name), img_name))
            os.mkdir(tmp)

            for i,image in enumerate(perturbed_images[img_name]):

                name = '{}_{}.png'.format(perturbation, i)
                image.save(os.path.join(tmp, name))

        
    else: # open the images
        
        perturbed_images = {}
        
        for img_name in source_img_names:

            tmp_dir = os.path.join(os.path.join(target_dir, directory_name), img_name)
            perturbed_images[img_name] = [Image.open(os.path.join(tmp_dir, im)).convert('RGB') for im in os.listdir(tmp_dir)]

    return perturbed_images


def load_image(source, name, preprocessing = None):
    """
    loads and returns a list of edited images
    """

    if preprocessing is None:
        preprocessing = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        ])

    image = preprocessing(
        Image.open(os.path.join(source,name)).convert('RGB')
    )

    return image
    

def retrieve_edited_samples(path, name, preprocessing = None):
    """
    retrieves the imagenet-E samples
    returns a list of PIL.Images with a 
    given preprocessing
    If preprocessing is none, a standard ImageNet preprocessing
    is applied
    """

    if preprocessing is None:
        preprocessing = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
    ])
        
    name = name[:-5]

    # get the directories
    kinds = [k for k in os.listdir(path) if not k in ['vis.py', 'labels.txt', 'eval.py', 'ori', 'full']]
    raw = ['ori', 'full']


    images = [preprocessing(Image.open(os.path.join(path, os.path.join(k, '{}.png'.format(name)))).convert('RGB')) for k in kinds]
    for r in raw:
        images.append(
            preprocessing(Image.open(os.path.join(path, os.path.join(r, '{}.JPEG'.format(name)))).convert('RGB'))
        )

    return images


def compute_wcam_on_source_and_altered_samples(model, img_names, samples, explainer, perturbation, opt):
    """
    computes the predictions and the explanations on the predictions
    given a model and a sample

    img_names : the name of the samples in the sample tensor
    samples : a list of PIL images
    explainer : a WaveletSobol instance
    perturbation (str) : 'corruption' or 'editing'
    opt : the dictionnary of parameters

    ## ! ## For editing, we retrieve the samples from
    the dataset, no alteration is comptuted on the fly
    so the img_names list should correspond to images
    that are contained in the dataset


    returns a dictionnary with filtered items :
    {img_name : 
            {
            source : (wcam, spatial_wcam),
            target : [(wcam, spatial_wcam)]
            }
    }

    and the dictionnary of the predictions
    {img_name : 
            {
            source : pred,
            target : [preds]
            }
    }
    """

    batch_size = opt['batch_size']
    imagenet_e_directory = opt['imagenet_e_directory']
    normalize = opt['normalization'] # a torchvision.Compose normalization

    # compute the predictions on the source and target samples

    x = torch.stack([
        normalize(im) for im in samples
    ])

    # compute the target corruptions 
    if perturbation == "corruptions":
        corrupted_samples = [corrupt_image(im) for im in samples]
    elif perturbation == "editing":
        corrupted_samples = [retrieve_edited_samples(imagenet_e_directory, img_name) for img_name in img_names]

    # corrupted_samples is a list of lists
    # where each item corresponds to a sequence of perturbations 
    # of one input image
    # transform the corrupted_samples as a list of stacked tensors
    corrupted_samples = [
        torch.stack([normalize(im) for im in subset]) for subset in corrupted_samples
    ]

    preds_source = evaluate_model_on_samples(x, model, batch_size)

    preds = {img_name : {} for img_name in img_names}

    for i, img_name in enumerate(list(preds.keys())):

        # baseline prediction:
        preds[img_name]['source'] = preds_source[i]

        # vector of prediction on the altered samlpes
        preds_target = evaluate_model_on_samples(corrupted_samples[i], model, batch_size)
        preds[img_name]['target'] = preds_target

    # now that we've computed the predictions, we filter them to compute 
    # the wcam of those that are differents

    wcams = {}
    for i, img_name in enumerate(preds.keys()):

        # list of images and labels to explain
        images_to_explain = []
        preds_to_explain = []

        # source image and label
        images_to_explain.append(normalize(samples[i]))
        preds_to_explain.append(
            preds[img_name]['source']
        )

        # get the corrupted samples for this image
        tmp_corrupted = corrupted_samples[i]

        # now filter the predictions on the target domain to select
        # the corrupted images from which we will compute the prediction

        altered_indices = np.where(
            preds[img_name]['target'] != preds[img_name]['source']
        )[0]

        if len(altered_indices) == 0: # pass if the model has not been affected by the 
                                      # corruptions for this sample
            continue
        else:
            # add the corresponding images to the list 
            # of samples to explain
            for altered_index in altered_indices:

                images_to_explain.append(tmp_corrupted[altered_index])
                preds_to_explain.append(
                    preds[img_name]['target'][altered_index]
                )

            # prepare the data
            x = torch.stack(
                images_to_explain
            )
            y = np.array(preds_to_explain).astype(np.uint8)

            # compute the explanations
            explanations = explainer(x,y)

            # add the explanations to the dictionnary

            wcams[img_name] = {
                'source' : (explanations[0], explainer.spatial_cam[0]),
                'target' : [(wcam, spatial_wcam) for wcam, spatial_wcam in zip(explanations[1:], explainer.spatial_cam[1:])] 
            }

    # returns
    return wcams, preds


def postprocess(explanations, spatial_cams, destination, images_data):
    """
    save the generated data under the following structure

    target_dir/img_name/
                - wcam_source
                - spatial_wcam_source
                - wcam_target_1
                - spatial_wcam_target_1.png
                - ...
    """
    def NormalizeData(data):
        """helper to normalize in [0,1] for the plots"""
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def normalize_and_save(wcam, spatial_wcam, destination, label):
        """
        saves the wcam and the spatial wcam
        """
        normalized_wcam = NormalizeData(wcam)
        normalized_swcam = NormalizeData(spatial_wcam)

        img_wcam = Image.fromarray((normalized_wcam * 255).astype(np.uint8))
        img_swcam = Image.fromarray((normalized_swcam * 255).astype(np.uint8))

        img_wcam.save(os.path.join(destination, 'wcam_{}.png'.format(label)))
        img_swcam.save(os.path.join(destination, 'spatial_wcam_{}.png'.format(label)))      


    for i, _ in enumerate(images_data):

        if i == 0:
            label = 'source'
        else:
            label = 'target_{}'.format(i)

        wcam = explanations[i]
        spatial_cam = spatial_cams[i]
        normalize_and_save(wcam, spatial_cam, destination, label)

 #    # set up the directory
 #    # img_names = list(wcams.keys())
# 
 #    # save the wcams
 #    for img_name in img_names:
# 
 #        destination = os.path.join(target_dir, img_name)
 #        if not os.path.exists(destination):
 #            os.mkdir(destination)
# 
 #        # retrieve the wcam
 #        source_wcam, source_spatial_wcam = wcams[img_name]['source']
 #        # export the wcam
 #        normalize_and_save(source_wcam, source_spatial_wcam, destination, 'source')
# 
 #        # retrive the target_wcams
 #        for i, items in enumerate(wcams[img_name]['target']):
 #            target_wcam, target_spatial_wcam = items
 #            normalize_and_save(target_wcam, target_spatial_wcam, destination, 'target_{}'.format(i))
# 
 #    # save the preds
 #    # with open(os.path.join(target_dir, 'preds.json'), 'w') as f:
 #    #    json.dump(preds, f, cls = NpEncoder)
# 
# now compute the distance between 
def compute_distance_between_wcams(wcams, labels, classes, distance = 'wasserstein', sample = 100):
    """
    returns the distance between the input wcam with a label label
    and wcams from the classes dictionnary
    """

    if distance == 'wasserstein':
        dist = wasserstein_distance_2d
    elif distance == 'euclidean':
        dist = frobenius_distance_2d
    elif distance == 'mmd':
        dist = mmd_distance

    # distances = np.zeros((len(wcams), 2))
    ranks = []

    for i, (wcam, label) in enumerate(zip(wcams, labels)):

        # get the wcam corresponding to the class
        class_wcam = classes[label]

        # randomly sample another label and retrieve the 
        remaining_labels = [l for l in classes.keys() if not l == label]
        rls = np.random.choice(remaining_labels, size = sample)

        random_dists = []
        for rl in rls:
            random_wcam = classes[rl]
            random_dists.append(dist(wcam, random_wcam))

        ref_dist = dist(wcam, class_wcam)
        random_dists.append(ref_dist)
        sorted_list = np.sort(random_dists)
        ranks.append(list(sorted_list).index(ref_dist))
        # we have a set of distances and the true distance
        # compute the rank
        # distances[i,0] = dist(wcam, class_wcam)
        # distances[i,1] = dist(wcam, random_wcam)

    return ranks

def wasserstein_distance_2d(a, b):
    """
    Computes the Wasserstein distance between two 2D numpy arrays.

    Parameters:
    a (numpy.ndarray): The first 2D numpy array.
    b (numpy.ndarray): The second 2D numpy array.

    Returns:
    float: The Wasserstein distance between the two arrays.
    """

    # Reshape the arrays into 1D arrays
    a = a.flatten()
    b = b.flatten()

    # Normalize the arrays to make them probability distributions
    a = a / np.sum(a)
    b = b / np.sum(b)

    # Compute the Wasserstein distance between the two distributions
    w_dist = wasserstein_distance(a, b)

    return w_dist

def frobenius_distance_2d(a, b):
    """
    Computes the Frobenius distance between two 2D numpy arrays.

    Parameters:
    a (numpy.ndarray): The first 2D numpy array.
    b (numpy.ndarray): The second 2D numpy array.

    Returns:
    float: The Frobenius distance between the two arrays.
    """

    # Compute the element-wise difference between the two arrays
    diff = a - b

    # Compute the Euclidean norm of the difference array
    frob_dist = np.linalg.norm(diff)

    return frob_dist

def mmd_distance(a, b, kernel_type='laplacian', gamma=None):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two 2D numpy arrays.

    Parameters:
    a (numpy.ndarray): The first 2D numpy array.
    b (numpy.ndarray): The second 2D numpy array.
    kernel_type (str): The type of kernel to use for computing the MMD. Can be either 'gaussian' or 'laplacian'.
    gamma (float): The gamma parameter for the Gaussian or Laplacian kernel. If None, the gamma parameter will be
                   automatically set based on the median distance between the samples.

    Returns:
    float: The MMD between the two arrays.
    """

    if kernel_type == 'gaussian':
        kernel_func = gaussian_kernel
    elif kernel_type == 'laplacian':
        kernel_func = laplacian_kernel
    else:
        raise ValueError('Invalid kernel type. Must be either "gaussian" or "laplacian".')

    # Compute the kernel matrices for the two arrays
    K_aa = kernel_func(a, a, gamma)
    K_ab = kernel_func(a, b, gamma)
    K_bb = kernel_func(b, b, gamma)

    # Compute the MMD
    mmd = np.mean(K_aa) + np.mean(K_bb) - 2 * np.mean(K_ab)

    return mmd


def gaussian_kernel(x, y, gamma=None):
    """
    Computes the Gaussian kernel between two 2D numpy arrays.

    Parameters:
    x (numpy.ndarray): The first 2D numpy array.
    y (numpy.ndarray): The second 2D numpy array.
    gamma (float): The gamma parameter for the Gaussian kernel. If None, the gamma parameter will be
                   automatically set based on the median distance between the samples.

    Returns:
    numpy.ndarray: The kernel matrix.
    """

    if gamma is None:
        gamma = 1 / np.median(pairwise_distances(x, y))

    pairwise_dists = pairwise_distances(x, y)
    kernel_matrix = np.exp(-gamma * pairwise_dists ** 2)

    return kernel_matrix


def laplacian_kernel(x, y, gamma=None):
    """
    Computes the Laplacian kernel between two 2D numpy arrays.

    Parameters:
    x (numpy.ndarray): The first 2D numpy array.
    y (numpy.ndarray): The second 2D numpy array.
    gamma (float): The gamma parameter for the Laplacian kernel. If None, the gamma parameter will be
                   automatically set based on the median distance between the samples.

    Returns:
    numpy.ndarray: The kernel matrix.
    """

    if gamma is None:
        gamma = 1 / np.median(pairwise_distances(x, y))

    pairwise_dists = pairwise_distances(x, y)
    kernel_matrix = np.exp(-gamma * pairwise_dists)

    return kernel_matrix


def pairwise_distances(x, y):
    """
    Computes the pairwise Euclidean distances between two numpy arrays.

    Parameters:
    x (numpy.ndarray): The first numpy array.
    y (numpy.ndarray): The second numpy array.

    Returns:
    numpy.ndarray: The pairwise distances matrix.
    """

    n_x = x.shape[0]
    n_y = y.shape[0]

    dist_matrix = np.zeros((n_x, n_y))

    for i in range(n_x):
        for j in range(n_y):
            dist_matrix[i, j] = np.sqrt(np.sum((x[i] - y[j]) ** 2))

    return dist_matrix




def compute_spectral_profile(item_names, label, opt):
    """
    computes the spectral profile based on the items in item names
    and the model passed as input

    returns a np.ndarray corresponding to the spectral profile
    """
    # retrive the parameters
    source_dir = opt['source_dir']
    preprocessing = opt['preprocessing']
    normalize = opt['normalize']
    explainer = opt['explainer']

    # load and set up the images
    images = [preprocessing(Image.open(os.path.join(source_dir, item)).convert('RGB')) for item in item_names]

    x = torch.stack([
        normalize(im) for im in images
    ])

    y = (label * np.ones(len(item_names))).astype(np.uint8)

    # explainer
    explanations = explainer(x,y)

    # now that the explanations are computed, average them and return the average
    return np.mean(np.array(explanations), axis = 0)

def reshape_wcam(arr, grid_size, levels):
    """
    computes the average importance across each location 
    of the wavelet transform

    return coeff, an array of shape (levels * 3) that contains 
    the importance of each coeffs 
    """

    coeffs = []

    for level in range(levels):

        # split the array by type of coefficient
        start, end = grid_size // 2 ** (level + 1), grid_size // 2 ** level
        shape = (end - start) ** 2 

        # sum the values for the horizontal, vertical
        # and diagonal coefficients
        if not level == levels - 1:

            shape = (end - start) ** 2 
            h = np.sum(arr[:start, start:end]) / shape
            d = np.sum(arr[start:end, start:end]) / shape
            v = np.sum(arr[start:end, :start]) /shape
    
            # sort them and append to the list
            coeffs.append([h,d,v])
        else:

            d = np.sum(arr[:end, :end]) / shape
            coeffs.append([d])


    # reverse the list to get the lowest frequencies first
    coeffs = coeffs[::-1]

    # flatten the list and return it
    return np.array(list(sum(coeffs, [])))

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



def compute_robustness_and_depth(source_img_names, imagenet_dir, imagenet_e_directory, target_dir, labels_true, wavelet, model, batch_size = 128, perturbation = None):
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
    - imagenet_e_dir : the directory where the edited images are stored
    - labels_true : a pd.DataFrame of labels
    - wavelet : the WaveletSobol explainer

    returns : 
    - reconstruction depth: a dictoinnary with the reconstruction depths (integers) 
                            keys corresponds to the cases 'source' or 'corrupted'
                            also contains the robustness, a list with the robustnesses 
                            (comprised between 0 and 1)

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
#        torchvision.transforms.Resize(256),
#        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize,
    ])

    # source images
    source_images = [Image.open(os.path.join(imagenet_dir, img_name)).convert('RGB') for img_name in source_img_names]
    source_images = [resize_and_crop(im) for im in source_images]

    # source predictions 
    x_source = torch.stack([
        preprocessing(im) for im in source_images
    ])

    source_preds = evaluate_model_on_samples(x_source, model, batch_size)

    # open or initialize the samples list
    perturbed_images = initialize_samples(source_img_names, imagenet_dir, imagenet_e_directory, target_dir, perturbation = perturbation)

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
        label = source_preds[i]

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


    # add the robustness as an additional key
    reconstruction_depth['robustness'] = robustness

    return reconstruction_depth


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

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # 👇️ alternatively use str()
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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