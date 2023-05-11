from math import ceil

import cv2
import numpy as np
import tensorflow as tf

from .estimators import JansenEstimator
from .sampling import ScipySobolSequence, TFSobolSequence
from .tf_perturbations import inpainting, wavelet, _wavelet
from .utils import *

#- this version: 30/03/2023 - 15:30 -#

class WaveletSobol:
    """
        An adaptation of the class SobolAttributionMethod to wavelet perturbations
        to the image

        Perturbations are the same grid as those defined in the baseline class. The only difference
        is that the class activation map `sti` are to be interpreted in the wavelet domain
        and not the image domain.

        Parameters
        ----------
        grid_size: int, optional
            Cut the image in a grid of grid_size*grid_size to estimate an indice per cell.
        nb_design: int, optional
            Must be a power of two. Number of design, the number of forward will be nb_design(grid_size**2+2).
        sampler : Sampler, optional
            Sampler used to generate the (quasi-)monte carlo samples.
        estimator: Estimator, optional
            Estimator used to compute the total order sobol' indices.
        perturbation_function: function, optional
            Function to call to apply the perturbation on the input.
        batch_size: int, optional,
            Batch size to use for the forwards.

        """

    def __init__(
        self,
        model,
        grid_size=8,
        nb_design=64,
        sampler=ScipySobolSequence(scramble = True),
        estimator=JansenEstimator(),
        #perturbation_function=wavelet,
        batch_size=256,
        levels=3,
        opt = None
    ):

        assert (nb_design & (nb_design-1) == 0) and nb_design != 0,\
            "The number of design must be a power of two."

        self.model = model

        self.grid_size = grid_size
        self.nb_design = nb_design
        #self.perturbation_function = perturbation_function

        self.sampler = sampler
        self.estimator = estimator

        self.batch_size = batch_size

        self.opt = opt # speciefies advanced sampling options

        self.levels = levels

        if self.opt is None or 'sampling' not in self.opt.keys():
            # default sampling method: mask that is uniformly applied
            # across the wt of the image
            masks = sampler(grid_size**2, nb_design)

        elif 'sampling' in self.opt.keys() and self.opt['sampling'] == "diagonal":
            # if the sampling is diagonal, then at each scale the same
            # mask is applied. We are therefore not interested in assessing
            # the impact of the vertical and horizontal coefficients at the
            # different scales
            # approximatively lowers the number of sequences by
            # 65%
            coeff_split = [(((2 ** k) * grid_size) ** 2) for k in range(levels)]
            coeff_split.insert(0, (grid_size ** 2)) # number of coeffs per level
            coeff_count = sum(coeff_split)
            
            masks = sampler(coeff_count, nb_design)

        #masks = sampler(grid_size**2 * (1 + 3 * levels), nb_design)#.reshape((-1, 1, grid_size, grid_size))
        self.masks = masks  # torch.Tensor(masks).cuda()
        # self.masks = masks.reshape((-1, 1, grid_size, grid_size))


    def __call__(self, inputs, labels):
        """
        Explain a particular prediction

        Parameters
        ----------
        inputs: ndarray or TensorFlow.Tensor [Nb_samples, Width, Height, Channels]
            Images to explain.
        labels: list of int,
            Label of the class to explain.
        """
        input_shape = inputs.shape[1:-1]
        explanations = []

        if self.opt is None or 'sampling' not in self.opt.keys():
            reshaped_masks = [
                cv2.resize(self.masks[i,:].reshape(self.grid_size,self.grid_size), input_shape, interpolation=cv2.INTER_NEAREST) for i in range(self.masks.shape[0])
                ]

        elif 'sampling' in self.opt.keys() and self.opt['sampling'] == "diagonal":
            reshaped_masks = expand_masks(self.masks, self.grid_size, self.levels, input_shape)

        # if specified so, set the first coefficients of the sequence to 0
        # to remove the impact of the approximation coefficients
        if self.opt is not None:
            if 'approximation' in self.opt.keys():
                if self.opt['approximation'] == False:
                    # reshape the reshaped masks to set the
                    # indices corresponding to the approximation
                    # to 0
                    limit = input_shape[0] // (2 ** (self.levels))
                    for mask in reshaped_masks:
                        mask[:limit, :limit] = 0

        # initialize the lists that are returned at the end
        self.spatial_cam = []

        for input, label in zip(inputs, labels):

            #perturbator = self.perturbation_function(input, reshaped_masks)
            wavelet_transform = compute_wavelet_transform(input, level=self.levels)

            y = np.zeros((len(self.masks)))
            nb_batch = ceil(len(self.masks) / self.batch_size)

            for batch_index in tf.range(nb_batch):
                # retrieve masks of the current batch
                start_index = batch_index * self.batch_size
                end_index = min(len(self.masks), (batch_index+1)*self.batch_size)

                batch_masks = reshaped_masks[start_index:end_index]

                # apply perturbation to the input and forward
                perturbated_inputs = _wavelet(wavelet_transform, batch_masks)

                batch_y = self.model(perturbated_inputs)

                # store the results
                batch_y = batch_y[:, label]
                y[start_index:end_index] = batch_y.numpy()

            # get the total sobol indices
            sti = self.estimator(self.masks, y, self.nb_design)
            #sti = resize(sti[0], input_shape)

            # project the sobol total indices on the spectrum
            if self.opt is None or 'sampling' not in self.opt.keys():
                # reshape to the size of the grid and upsample to the size of the input
                sti_raw = sti.reshape((self.grid_size, self.grid_size))
                sti_mask = cv2.resize(sti_raw, input_shape, interpolation = cv2.INTER_CUBIC)
                sti_spatial = reproject(sti_mask, input_shape, self.levels)

                # return the raw or reshaped coefficients
                if self.opt is not None and 'size' in self.opt.keys():

                    if self.opt['size'] != self.grid_size:
                        size = (self.opt['size'], self.opt['size'])
                        sti_mask = cv2.resize(sti_raw, size, interpolation = cv2.INTER_CUBIC)
                        explanations.append(sti_mask)
                    else:
                        explanations.append(sti_raw)
                else:
                    explanations.append(sti_mask)

            elif 'sampling' in self.opt.keys() and self.opt['sampling'] == "diagonal":

                sti_components = wrap_and_upscale(sti, self.grid_size, input_shape, self.levels, interpolation = cv2.INTER_CUBIC)
                sti_mask = convert_as_mask(sti_components, input_shape, self.levels,self.opt)
                sti_spatial = reproject(sti_components, input_shape, self.levels)

                # return the main object: the cam in the wavelet domain
                explanations.append(sti_mask)

            # return the spatial cam 
            self.spatial_cam.append(sti_spatial)

        return explanations

class SobolAttributionMethod:
    """
    Sobol' Attribution Method.

    Once the explainer is initialized, you can call it with an array of inputs and labels (int) 
    to get the STi.

    Parameters
    ----------
    grid_size: int, optional
        Cut the image in a grid of grid_size*grid_size to estimate an indice per cell.
    nb_design: int, optional
        Must be a power of two. Number of design, the number of forward will be nb_design(grid_size**2+2).
    sampler : Sampler, optional
        Sampler used to generate the (quasi-)monte carlo samples.
    estimator: Estimator, optional
        Estimator used to compute the total order sobol' indices.
    perturbation_function: function, optional
        Function to call to apply the perturbation on the input.
    batch_size: int, optional,
        Batch size to use for the forwards.
    """

    def __init__(
        self,
        model,
        grid_size=8,
        nb_design=64,
        sampler=TFSobolSequence(),
        estimator=JansenEstimator(),
        perturbation_function=inpainting,
        batch_size=256
    ):

        assert (nb_design & (nb_design-1) == 0) and nb_design != 0,\
            "The number of design must be a power of two."

        self.model = model

        self.grid_size = grid_size
        self.nb_design = nb_design
        self.perturbation_function = perturbation_function

        self.sampler = sampler
        self.estimator = estimator

        self.batch_size = batch_size

        self.masks = sampler(grid_size**2, nb_design).reshape((-1, grid_size, grid_size, 1))

    def __call__(self, inputs, labels):
        """
        Explain a particular prediction

        Parameters
        ----------
        inputs: ndarray or tf.Tensor [Nb_samples, Width, Height, Channels]
            Images to explain.
        labels: list of int,
            Label of the class to explain.
        """
        input_shape = inputs.shape[1:-1]
        explanations = []

        for input, label in zip(inputs, labels):

            perturbator = self.perturbation_function(input)

            y = np.zeros((len(self.masks)))
            nb_batch = ceil(len(self.masks) / self.batch_size)

            for batch_index in range(nb_batch):
                # retrieve masks of the current batch
                start_index = batch_index * self.batch_size
                end_index = min(len(self.masks), (batch_index+1)*self.batch_size)
                batch_masks = self.masks[start_index:end_index]

                # apply perturbation to the input and forward
                batch_y = SobolAttributionMethod._batch_forward(self.model, input, batch_masks,
                                                                perturbator, input_shape)

                # store the results
                batch_y = batch_y[:, label].numpy()
                y[start_index:end_index] = batch_y

            # get the total sobol indices
            sti = self.estimator(self.masks, y, self.nb_design)
            sti = resize(sti, input_shape)

            explanations.append(sti)

        return np.array(explanations)

    @staticmethod
    @tf.function
    def _batch_forward(model, input, masks, perturbator, input_shape):
        upsampled_masks = tf.image.resize(masks, input_shape)
        perturbated_inputs = perturbator(upsampled_masks)
        outputs = model(perturbated_inputs)
        return outputs
