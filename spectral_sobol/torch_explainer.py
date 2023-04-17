from math import ceil
import numpy as np
import torch
from torch.nn.functional import interpolate
import torchvision
from .estimators import JansenEstimator
from .sampling import ScipySobolSequence
from .torch_perturbations import inpainting, wavelet, fourier, _fourier
from .utils import *
import cv2
import torch.nn.functional as F

class FourierSobol:
    """
    An adaptation of the original Sobol attribution method
    for Fourier class activation maps. 

    You can specify the type of perturbation (grid, square, circle):

    - 'grid' works like the original Sobol attribution method, but with perturbations
      in the Fourier amplitude spectrum instead of the image space (like Chen et al),
    - 'square' applies square masks in the Fourier amplitude spectrum of the image to 
       compute the perturbations (similar to Zhang et al),
    - 'circle' applies circular masks in the Fourier amplitude spectrum of the image

    We recommend you use circle perturbations as they are less likely to generate 
    reconstruction artifacts (ripplies) during reconstruction.

    For the square and circle perturbation, the STi are unidimensional and correspond
    to the weight of each component.

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
    perturbation : (str) the type of perturbation
    """

    def __init__(
        self,
        model,
        grid_size=8,
        nb_design=64,
        sampler=ScipySobolSequence(scramble=True),
        estimator=JansenEstimator(),
        perturbation_function=fourier,
        batch_size=256,
        perturbation = 'circle',
        opt = None
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

        self.perturbation = perturbation

        if perturbation == "grid":
            masks = sampler(grid_size**2, nb_design)

        else: 
            masks = sampler(grid_size, nb_design)

        self.masks = masks
        self.opt = opt

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
        input_shape = inputs.shape[2:]
        explanations = []
        reshaped_masks = generate_fourier_masks(self.masks,input_shape, self.grid_size, perturbation=self.perturbation)

        # self.dispersion = []

        for input, label in zip(inputs, labels):


            perturbator = self.perturbation_function(input, reshaped_masks)
            fourier_transform = compute_fourier_transform(input.cpu().permute(1,2,0))


            y = np.zeros((len(self.masks)))
            nb_batch = ceil(len(self.masks) / self.batch_size)

            #self.transformed_images = []

            for batch_index in range(nb_batch):
                # retrieve masks of the current batch
                start_index = batch_index * self.batch_size
                end_index = min(len(self.masks), (batch_index+1)*self.batch_size)
                batch_masks = reshaped_masks[start_index:end_index]

                # apply perturbation to the input and forward
                batch_y = FourierSobol._batch_forward(self.model, fourier_transform, batch_masks,
                                                                perturbator, input_shape)

                # store the results
                batch_y = batch_y[:, label].cpu().detach().numpy()
                y[start_index:end_index] = batch_y

            # get the total sobol indices
            sti_components = self.estimator(self.masks, y, self.nb_design)

            if self.opt is not None and 'signed' in self.opt.keys():

                if self.opt['signed'] == True:
                    # also return the signed components
                    baseline_pred = self.model(input.unsqueeze(0)).cpu().detach().numpy()

                    altered_preds = FourierSobol._remove_component(self.model, self.perturbation, input_shape, fourier_transform, self.grid_size, self.batch_size, label)
                    
                    signs = baseline_pred[:,label] - altered_preds
                    # convert the signs as a vector of +/- 1 
                    signs = 2 * (signs > 0).astype(int) - 1
                    sti_components *= signs

            if self.opt is not None and 'resize' in self.opt.keys():
                if self.opt['resize'] == False: # keep the shape of the grid size feature map if asked to do so
                    sti = fourier_projection(sti_components, (self.grid_size, self.grid_size), self.grid_size, self.perturbation)
                else:
                    sti = fourier_projection(sti_components, input_shape, self.grid_size, self.perturbation)


            explanations.append(sti)


        return explanations
    

    @staticmethod
    def _remove_component(model, perturbation, input_shape, fourier_transform, grid_size, batch_size, label):
        """
        computes f(x_{\i}) where i is the ith component and x is the 
        input image

        returns a vector of the shape grid_size components and where each component
        is the prediction of f for input x without the ith component

        """
        # generate the masks depending on the case

        if perturbation == 'circle':
            masks_bank = generate_circular_masks(input_shape, grid_size)
            masks = [1 - masks_bank[:,:,i] for i in range(masks_bank.shape[-1])]


        elif perturbation == 'square':
            masks_bank = generate_square_masks(input_shape, grid_size)
            masks = [1 - masks_bank[:,:,i] for i in range(masks_bank.shape[-1])]

        elif perturbation == "grid":

            masks = []
            for i in range(grid_size ** 2):
                tmp = np.ones(grid_size ** 2)
                tmp[i] -= 1 # we remove the ith component
                mask = cv2.resize(tmp.reshape(grid_size, grid_size), input_shape, interpolation = cv2.INTER_NEAREST)
                masks.append(mask) # add the mask upsampled to the dimensions of the input

        # compute the perturbated inputs
        perturbed_inputs = _fourier(fourier_transform, masks)

        # prepare the inputs for inference
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
        ])

        perturbed_images = torch.stack([transforms(im) for im in perturbed_inputs])
        device = next(model.parameters()).device

        # outpout predictions
        y = np.zeros((len(masks)))


        if perturbed_images.shape[0] > batch_size:

            # do inference per batch
            nb_batch = ceil(len(masks) / batch_size)

            #self.transformed_images = []

            for batch_index in range(nb_batch):
                # retrieve masks of the current batch
                start_index = batch_index * batch_size
                end_index = min(len(masks), (batch_index+1)*batch_size)

                with torch.no_grad():
                    # predictions for the current batch 
                    batch_preds = model(perturbed_images[start_index:end_index,:,:,:].to(device))

                # store the results
                batch_y = batch_preds[:, label].cpu().detach().numpy()
                y[start_index:end_index] = batch_y
    
        else:
            # single pass otherwise
            outputs = model(perturbed_images.to(device))
            y = outputs[:,label].cpu().detach().numpy()

        return y

    @staticmethod
    def _batch_forward(model, fourier_transform, masks, perturbator, input_shape):


        # define the in transform and out transforms for the images
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
        ])

        # computes the perturbed inputs 
        perturbated_inputs = perturbator(fourier_transform, masks)

        device = next(model.parameters()).device

        # transform the perturbated input as a tensor
        perturbed_images = torch.stack([transforms(im) for im in perturbated_inputs]).to(device)
        outputs = model(perturbed_images)
        return outputs

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
        sampler=ScipySobolSequence(scramble=True),
        estimator=JansenEstimator(),
        perturbation_function=wavelet,
        batch_size=256,
        levels=3,
        opt = None
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
        inputs: ndarray or PyTorch.Tensor [Nb_samples, Width, Height, Channels]
            Images to explain.
        labels: list of int,
            Label of the class to explain.
        """
        input_shape = inputs.shape[2:]
        explanations = []

        if self.opt is None or 'sampling' not in self.opt.keys():
        #    reshaped_masks = expand_masks(self.masks, self.grid_size, self.levels, input_shape)
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

                    # update the value of the coefficients
                    for mask in reshaped_masks:
                        mask[:limit,:limit] = 0

        # initialize the lists that are returned at the end

        self.spectral_spread = []
        self.spatial_spread = []
        self.spatial_cam = []

        for input, label in zip(inputs, labels):


            perturbator = self.perturbation_function(input, reshaped_masks)
            wavelet_transform = compute_wavelet_transform(input.cpu().permute(1,2,0), level = self.levels)


            y = np.zeros((len(self.masks)))
            nb_batch = ceil(len(self.masks) / self.batch_size)

            for batch_index in range(nb_batch):
                # retrieve masks of the current batch
                start_index = batch_index * self.batch_size
                end_index = min(len(self.masks), (batch_index+1)*self.batch_size)
                #batch_masks = self.masks[start_index:end_index]
                batch_masks = reshaped_masks[start_index:end_index]
                #resized_batch = duplicate_masks(batch_masks, self.grid_size, self.levels, input_shape)

                # apply perturbation to the input and forward
                batch_y = WaveletSobol._batch_forward(self.model, wavelet_transform, batch_masks,
                                                                perturbator, input_shape)

                # store the results
                batch_y = batch_y[:, label].cpu().detach().numpy()
                y[start_index:end_index] = batch_y

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

            # compute the spectral dispersion and the robustness margin
            robustness_margin = clustered_entropy(sti_mask, self.levels)
            spatial_spread = compute_spatial_spread(sti_spatial)

            self.spectral_spread.append(robustness_margin)
            self.spatial_spread.append(spatial_spread)

        return explanations

    @staticmethod
    def _batch_forward(model, wavelet_transform, masks, perturbator, input_shape):

        # define the in transform and out transforms for the images
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
        ])

        # computes the perturbed inputs 
        perturbated_inputs = perturbator(wavelet_transform, masks)

        device = next(model.parameters()).device
        # transform the perturbated input as a tensor and send them to the correct device
        perturbed_images = torch.stack([transforms(im) for im in perturbated_inputs]).to(device)
        outputs = model(perturbed_images)

        return outputs

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
        sampler=ScipySobolSequence(),
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

        device = next(self.model.parameters()).device

        masks = sampler(grid_size**2, nb_design).reshape((-1, 1, grid_size, grid_size))
        self.masks = torch.Tensor(masks).to(device)

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
        input_shape = inputs.shape[2:]
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
                batch_y = batch_y[:, label].cpu().detach().numpy()
                y[start_index:end_index] = batch_y

            # get the total sobol indices
            sti = self.estimator(self.masks, y, self.nb_design)
            sti = resize(sti[0], input_shape)

            explanations.append(sti)

        return explanations

    @staticmethod
    def _batch_forward(model, input, masks, perturbator, input_shape):
        upsampled_masks = interpolate(masks, input_shape)
        perturbated_inputs = perturbator(upsampled_masks)
        outputs = model(perturbated_inputs)
        return outputs
