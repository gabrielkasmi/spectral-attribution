import cv2
import numpy as np
import tensorflow as tf
from .utils import *
from PIL import Image


@tf.function
def _fourier(fourier_transform, masks):
    """
    pertrubs the magnitude spectrum
    of the input and inverts the spectrum
    to return the perturbed image
    """    

    perturbated_inputs = []

    for mask in masks:

        peturbed_image = inverse_perturbed_spectrum(fourier_transform, mask)

        perturbated_inputs.append(peturbed_image)

    return perturbated_inputs

def fourier(fourier_transform, masks):
    """
    returns f : callable
    """

    def f(fourier_transform, masks):
        return _fourier(fourier_transform, masks)

    return f

def _wavelet(wavelet_transform, masks, wavelet_type = "haar"):
    # compute the wavelet transform
    out, slices = wavelet_transform
    perturbed_inputs = []


    # for mask in masks.squeeze(1):
    for mask in masks:
        #reshape the masks to the size of the image
        # mask = cv2.resize(mask, out.shape[:2], interpolation = cv2.INTER_NEAREST)

        # compute the perturbation and append to the list
        pert = perturb_and_invert(mask, slices, out, wavelet = wavelet_type)
        img = Image.fromarray(pert)
        tensor = tf.convert_to_tensor(img)

        perturbed_inputs.append(tf.keras.applications.imagenet_utils.preprocess_input(tensor))

    return tf.convert_to_tensor(perturbed_inputs)

def wavelet(wavelet_transform, masks, wavelet = "haar"):
    """
    perturbs the input with the masks
    returns a list of altered images

    returns 

    f callable
    """
    
    def f(wavelet_transform, masks):
        return _wavelet(wavelet_transform, masks)
    return f


@tf.function
def _baseline_ponderation(x, masks, x0):
    return tf.expand_dims(x, 0) * masks + (1.0 - masks) * tf.expand_dims(x0, 0)


@tf.function
def _amplitude_operator(x, masks, sigma):
    return x[None, :, :, :] * (masks - 0.5) * sigma


def inpainting(input):
    """
    Tensorflow inpainting perturbation function.

    X_perturbed = X * M

    Parameters
    ----------
    input: tf.Tensor
        Image to perform perturbation on.

    Returns
    -------
    f: callable
        Inpainting perturbation function.
    """
    x0 = np.zeros(input.shape)
    x0 = tf.cast(x0, tf.float32)

    def f(masks):
        return _baseline_ponderation(input, masks, x0)
    return f


def blurring(input, sigma=10):
    """
    Tensorflow blur perturbation function.

    X_perturbed = blur(X, M)

    Parameters
    ----------
    input: tf.Tensor
        Image to perform perturbation on.
    sigma: int
        Blurring operator intensity.

    Returns
    -------
    f: callable
        Blur perturbation function.
    """
    x0 = cv2.blur(input.copy(), (sigma, sigma))
    x0 = tf.cast(x0, tf.float32)

    def f(masks):
        return _baseline_ponderation(input, masks, x0)
    return f


def amplitude(input, sigma=1.0):
    """
    Tensorflow amplitude perturbation function.

    X_perturbed = X + (M - 0.5) * sigma

    Parameters
    ----------
    input: tf.Tensor
        Image to perform perturbation on.
    sigma: int
        Amplitude operator intensity.

    Returns
    -------
    f: callable
        Blur perturbation function.
    """

    def f(masks):
        return _amplitude_operator(input, masks, sigma)
    return f
