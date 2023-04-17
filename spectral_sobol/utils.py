import cv2
import numpy as np
import pywt
from PIL import Image
#from scipy.stats import entropy


def define_regions(size, levels):
    """
    helper that splits the nyquist square into h,v,d regions
    at each level depending on the size of the input
    and the number of levels

    by convention, the approximation coefficients 
    have the highest value, which is then decreasing down 
    to the highest frequency detail coefficients
    """

    # define the number of regions : 3 per level
    # and 1 for the last approximation coefficients
    n_regions = (3 * levels) + 1

    # define the mask. Each region will be labelled
    # 1, ..., n_regions
    mask = n_regions * np.ones((size, size))

    # loop over the levels
    for l in range(levels):

        # define the labels for each detail coefficient
        # from each level
        offset = l * levels

        h, v, d = offset + 1, offset + 2, offset + 3

        # regions in the input that must be labelled
        start = int(size / 2 ** (l + 1))
        end = int(size / (2 ** l))

        # label the regions 
        mask[:end,start:end] = h
        mask[start:end,:end] = v
        mask[start:end, start:end] = d

    return mask

def spectral_redundancy(explanation, spatial_cam, size, grid_size, threshold = 0.95, levels = 3):
    """
    computes the spectral redundancy, which simulateneously summarizes
    the spread in the spatio-scale space

    the measure is defined as follows:
    1) computation of the local maxima in the nyquist square
    2) computation of the distance wrt the upper left corner (characterizes the scale spread) = alpha
    3) computation of the relative relative localizations (i.e. between 0 and 1 in each cell of the 
    wavelet transform)
    4) summation of the localizations on the spatial grid : 
        each cell (i,j) is += 1 / alpha if there is a maxima in the spot (i,j)
        and alpha is the inverse of the distance to the ul corner 
    at this step, we have a localization map weighted by the distance of the hotpots

    5) computation of the spatial spread (takes values in [1, dist_barycenter])
    6) aggregation : spectral redundancy = 1 / spatial_spread \times sum(coeffs(i,j))

    returns a scalar
    """

    # parameters
    width = grid_size // 2 ** (levels) # dimensions of the spatial map
    bins = [(i + 1) / width for i in range(width)] # bins to be used with the relative coordinates
    max_dist = np.sqrt(2) * np.sqrt(size ** 2 )

    # computation of the local maxima in the nyquist sqare
    maxima = compute_spatial_spread(explanation, threshold, scalar = False)
    # compute the spatial spread
    spatial_spread = compute_spatial_spread(spatial_cam)
    spatial_spread = max(1, spatial_spread) # rescale values equal to 0 to 1
    spatial_spread /= max_dist # normalize by the maximum spread

    # computation of the distances wrt the ul corner
    # of the wcam
    distances = np.linalg.norm(maxima, axis =1)

    # split the space into regions 
    regions = define_regions(size, levels)

    # redundancy map
    redundancy = np.zeros((width, width))

    for i, m in enumerate(maxima):

        # localize the local maxima
        value = regions[m[0], m[1]]
        
        # find the boundaries of the region
        coordinates = np.argwhere(regions == value)
        x_max, y_max = np.max(coordinates[:,0]), np.max(coordinates[:,1])

        # extract the coordinates of the 
        # local maxima 
        x, y = m[0], m[1]

        # normalize the coordinates
        x_norm = x / x_max
        y_norm = y / y_max

        # discretize the coordinates
        x_bin = np.digitize(x_norm, bins = bins)
        y_bin = np.digitize(y_norm, bins = bins)

        # increase the redundancy map by the inverse of the distance
        redundancy[x_bin, y_bin] += (1 / distances[i])

        # return the value of the redundancy index
        return (1 / spatial_spread) * sum(redundancy.flatten())

def spectral_spread(explanations, threshold = 0.95):
    """
    computes the spectral spread based on the WCAM
    and the quantile thresholding

    the spectral spread is defined as the distance between 
    the upper left corner and the barycenter + the distance between 
    the barycenter and its farthest point

    returns the scalar corresponding to the spread
    """
    # retrieve the maxima
    maxima = compute_spatial_spread(explanations, threshold, scalar = False)

    # compute the barycenter
    barycenter = np.mean(maxima, axis = 0)

    # compute the distance between the barycenter and the upper left 
    # corner of the image
    spread = np.sqrt(
        barycenter[0]**2 + barycenter[1]**2
    )

    # compute the distance between the barycenter and the other points
    points = len(maxima)
    barycenter = np.vstack([barycenter] * points)

    # compute the distance
    # consider the maximum distance and add it to the distance

    d = np.linalg.norm(barycenter-maxima, axis = 1)
    farthest = np.max(d)
    spread += farthest

    return spread


def compute_spatial_spread(map, quantile = 0.9):
    """
    computes the spatial spread of the spatial WCAM
    it corresponds to the distance between the two farthest maxima
    otherwise it is set to 0
    based on the quantile thresholding
    """
    maxima = get_maxima_map(map, quantile)
    return spatial_spread(maxima)

    

def return_local_maxima(contours, map):
    """
    returns the coordinates of the local maxima given 
    the contours passed as input.
    """

    maxima = []

    # fill the 
    for contour in contours:
        mask = np.zeros(map.shape)
        cv2.fillPoly(mask, pts =[contour], color=(255,255,255))
        mask /= 255.

        filtered = mask * map
        x, y = np.unravel_index(filtered.argmax(), map.shape)

        maxima.append(np.array([x,y]))
    

    return np.array(maxima)

def return_maxima_of_slice(slice, map):
    """
    given a slice, computes the contours and 
    returns the different local maxima
    """

    # compute the contours
    contours, _ = cv2.findContours(slice, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    return return_local_maxima(contours, map)

def slice_map(map, quantile):
    """
    returns the sliced mask of the map
    given the quantile passed as input
    """
    binaries = np.zeros(map.shape, dtype = np.uint8)
    q = np.quantile(map.flatten(), quantile)
    x, y = np.where(map > q)
    binaries[x,y] = 1

    return binaries

def get_maxima_map(map, quantile):
    """
    returns the coordinates of the maxima
    of the map, after quantile thresholding
    according to the quantile passed as input

    returns : a np.ndarray of coordinates
    """

    # threshold the map
    binary = slice_map(map, quantile)

    return return_maxima_of_slice(binary, map)


def spatial_spread(maxima):
    """
    given the array of maxima, 
    computes the center
    and the distance
    """

    center = np.mean(maxima, axis = 0)

    points = len(maxima)


    # if only one maxima, then the distance is 0
    if points == 0:
        return 0.
    else:
        # reshape the center to match the 
        # size of the maxima
        center = np.vstack([center] * points)

        # compute the distance
        d = np.linalg.norm(center-maxima, axis = 1)
        d = np.sort(d)[::-1]
        # return the cumulated distance of the two 
        # points that are the farthest from the center
        return np.sum(d[:2])

def clustered_entropy(sti_mask, levels):
    """
    computes the clustered entropy of the WCAM
    """
    size = sti_mask.shape[0]

    total_energy = np.sum(sti_mask.flatten())

    energies = []

    for i in range(levels):


        cutoff = int(size / (2 ** (i + 1)))
        subset = sti_mask[:cutoff, :cutoff]
        energies.append(np.sum(subset.flatten()))

    energies = np.array(energies) / total_energy
    shares = [1 - energies[0]]

    for i in range(energies.shape[0] - 1):
        shares.append(
            energies[i] - energies[i+1]
        )
    shares.append(energies[-1])

    shares = np.array(shares) * 100

    robustness_margin = shares[-1]
    #dispersion = entropy(shares)


    return robustness_margin #, dispersion 


    

def compute_robustness_margin(sti_mask):
    """
    computes the robustness margin of the prediction
    """
    # size of the mask
    size = sti_mask.shape[0]

    total_energy = np.sum(sti_mask.flatten())

    lf_part = sti_mask[:int(size/2), :int(size/2)]
    lf_energy = np.sum(lf_part.flatten())

    return (total_energy - lf_energy) / total_energy


def compute_fourier_transform(img):
    """
    computes the fourier transform
    returns the magnitude and phase of the spectrum

    args:
        img : a np.ndarray of the original image

    return :
        magnitude, phase (np.array), (np.array)
    the phase and the magnitude, for each channel of the input image
    """

    if not isinstance(img, np.ndarray):
        
        img = np.array(img)

    array_img = np.float32(img)


    spectrum = {}
    for i in range(img.shape[2]): #loop ove rthe channels
        dft = cv2.dft(array_img[:,:,i], flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        spectrum[i] = dft_shift

    return spectrum

def inverse_perturbed_spectrum(fft_image, mask):
    """
    given the spectrum, returns the inverse fourier transform
    """

    #plt.imshow(mask)
    #plt.show()

    w, h, _ = fft_image[0].shape # x and y dimensions of the output image

    # expand the mask to match the dimensions of the spectrum
    mask = np.array([mask, mask]).transpose(1,2,0)

    # create a new empty image
    img_filtered = np.zeros((w, h, 3))


    for i in range(3): #loop over the color channels

        # apply mask and inverse DFT
        fshift = fft_image[i] * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

        img_filtered[:,:,i] = img_back

    # convert the image as a PIL image

    #plt.imshow(img_filtered)
    #plt.show()

    img_filtered = Image.fromarray((NormalizeData(img_filtered) * 255).astype(np.uint8))


    return img_filtered

def generate_circular_masks(shape, grid_size):
    """
    computes a circular mask of bandwidth b computed from grid_size 
    with size shape (tuple)
    """
    
    d = int(shape[0]) # size of the input image
    center = int(shape[0] / 2), int(shape[1] / 2)
        
    Ms = {} # dictionnary, returned at the end, with the masks
    
    k = 0
    
    # distance in the diagonal between the center and the corner:
    d_max = np.sqrt(2) * np.sqrt(center[0] ** 2 + center[1] ** 2)
    
    # Compute the bandwidth b from grid_size
    # b = int(shape[0] / 2) / grid_size
    b = d_max / grid_size
    
    #d_current = np.sqrt((k * b) ** 2 + (k * b) ** 2)
    d_current = k * b
    
    while d_current < d_max:
        
        x, y = np.ogrid[:d, :d]
        x_center = x - center[0]
        y_center = y - center[1]
        
        r = np.sqrt(x_center ** 2 + y_center ** 2)
        M = (k * b <= r) & (r <= (k + 1) * b)
        
        Ms[k] = M.astype(int)
        
        # increment k and update the current distance
        k += 1
        #d_current = np.sqrt((k * b) ** 2 + (k * b) ** 2)
        d_current = k * b
        
    masks_aggregated = np.empty((224,224, len(Ms.keys())))

    for i,mask in enumerate(Ms.values()):
        
        masks_aggregated[:,:,i] = mask

    return masks_aggregated

def generate_square_masks(shape, grid_size):
    """
    computes a square mask of bandwidth b computed from grid_size 
    with size shape (tuple)
    """
    
    d = int(shape[0]) # size of the input image
    center = int(shape[0] / 2), int(shape[1] / 2)
        
    Ms = {} # dictionnary, returned at the end, with the masks
    
    k = 1
    
    # distance in the diagonal between the center and the corner:
    d_max = np.min(shape) / 2
    
    # Compute the bandwidth b from grid_size
    b = d_max / grid_size
    
    d_current = k * b
    
    while d_current <= d_max:
                
        M = np.zeros((d,d)) # create a mask of zeros
        # define the left and right coordinates of the mask
        left = int(center[0] - k * b)
        right = int(center[0] + k * b)
        
        # main loop that sets to 1 the coordinates
        # whose distance is within the bounds

        for x in range(left, right):
            for y in range(left, right):
                M[x, y] = 1
                
        Ms[k] = M
        
        # increment k and update the current distance
        k += 1
        d_current = k * b
        
    masks_aggregated = np.empty((224,224, len(Ms.keys())))

    for i, mask in enumerate(list(Ms.values())):
        
        masks_aggregated[:,:,i] = mask

    for i in list(Ms.keys())[1:]:
        masks_aggregated[:,:,i-1] -= Ms[i-1]

    return masks_aggregated

def generate_fourier_masks(masks, input_shape, grid_size, perturbation = 'circle'):
    """
    generates a list of masks to be applied to the fourier domain

    takes as input the unwrapped masks, wraps them according to the
    perturbation type and returns the list of masks reshaped to the size 
    of the input

    args:

    masks: a np.ndarray corresponding to the sobol sequence
    input_shape (tuple) : the size of the input
    grid_size (tuple) : the size of the grid
    perturbation (str) : the type of perturbation

    returns: 
    reshaped_mask : a list of masks reshaped to the size of the image (input_shape)
    """

    # initialize the mask bank for the circle or the squares
    if perturbation == 'circle':
        masks_bank = generate_circular_masks(input_shape, grid_size)

    elif perturbation == 'square':
        masks_bank = generate_square_masks(input_shape, grid_size)

    reshaped_masks = []

    for row in range(masks.shape[0]):

        sequence = masks[row,:]

        if perturbation == "grid":
            mask = cv2.resize(sequence.reshape(grid_size, grid_size), input_shape, interpolation = cv2.INTER_NEAREST)

        else:
            mask = np.dot(masks_bank, sequence)

        reshaped_masks.append(mask)

    return reshaped_masks

def fourier_projection(sti_components, input_shape, grid_size, perturbation):
    """
    reprojects the components in the fourier space

    either returns the grid or the weighted sum of the components
    """

    if perturbation == "grid":

        sti = cv2.resize(sti_components.reshape(grid_size, grid_size), input_shape, interpolation = cv2.INTER_CUBIC)
        return np.log(1 + sti)

    # initialize the mask bank for the circle or the squares
    if perturbation == 'circle':
        masks_bank = generate_circular_masks(input_shape, grid_size)

    elif perturbation == 'square':
        masks_bank = generate_square_masks(input_shape, grid_size)

    weighted_mask = np.dot(masks_bank, sti_components)

    return np.log(weighted_mask)

def wrap_and_upscale(sequence, grid_size, input_shape, levels, interpolation = cv2.INTER_NEAREST):
    """
    wraps a sequence as squared masks and upscale them to match the dimensions
    of the components of the wavelet transform of the input image

    args
    sequence: the (n,) np.ndarray corresponding to the sobol sequence
    grid_size (int) : the size of the grid
    input_shape (tuple) the shape of the input image
    the number of subsequences

    returns
    components: a dictionnary where each key correspond to a 
    subset of the WT of the image, starting from the low frequency (lf)
    to the vertical, horizontal and diagonal coefficients at the different
    levels

    each value is the mask for the corresponding level.
    """

    # define the cutoff, i.e. where to split the sequence
    # and the number of subsequences to extract from the main sequence
    cutoff = int(grid_size ** 2)
    subsequences = int((1 + 3  * levels))

    # for each sequece (i.e. line) generate masks
    # that will be applied to the cells of the wavelet transform
    thumbnails = [sequence[i * cutoff : (i+1)*cutoff].reshape(grid_size, grid_size) for i in range(subsequences)]

    # now map each thumbnail to a part of the WT
    # and upscale if necessary
    baseline_shape = max(grid_size, int(input_shape[0] / (2 ** levels)))

    components = {
        'lf' : cv2.resize(thumbnails[0], (baseline_shape, baseline_shape), interpolation = interpolation),
        'h1' : cv2.resize(thumbnails[1], (baseline_shape, baseline_shape), interpolation = interpolation),
        'd1' : cv2.resize(thumbnails[2], (baseline_shape, baseline_shape), interpolation = interpolation),
        'v1' : cv2.resize(thumbnails[3], (baseline_shape, baseline_shape), interpolation = interpolation)
    }

    # now consider the remaining levels, from the largest to the smallest
    for i, level in enumerate(range(1,levels)):
        size = (baseline_shape * (2 ** level), baseline_shape * (2 ** level))
        index = subsequences - 3 * (levels - (i + 1))

        components['h{}'.format(level + 1)] = cv2.resize(thumbnails[index], size, interpolation = interpolation)
        components['d{}'.format(level + 1)] = cv2.resize(thumbnails[index + 1], size, interpolation = interpolation)
        components['v{}'.format(level + 1)] = cv2.resize(thumbnails[index + 2], size, interpolation = interpolation)

    return components

def convert_as_mask(components, input_shape, levels, opt = None):
    """
    converts a dictionnary of components as a mask of the size
    of the input
    """

    if opt is not None:
        if 'size' in opt.keys():

            input_shape = (opt['size'], opt['size'])

    out = np.zeros(input_shape)

    # extract the mask corresponding to the low frequency components
    lf = components['lf']
    baseline_shape = lf.shape[0] # baseline shape that serves as a multiple
    
    out[:baseline_shape, :baseline_shape] = lf

    for level in range(levels):

        down, up = baseline_shape * (2 ** level), baseline_shape * (2 ** (level + 1))
        # vertical component
        out[: down, down : up] = np.rot90(components['v{}'.format(level + 1)], axes = (0,1))
        # diagonal component
        out[down : up, down : up] = components['d{}'.format(level + 1)]
        # horizontal component
        out[down : up, :down ] = np.rot90(components['h{}'.format(level + 1)], axes = (1,0))
        
    return out

def expand_masks(masks, grid_size, levels, input_shape):
    """
    convert the sequence as masks at different scales

    each cell of the wavelet transform of the image is covered by a set of coefficients
    for the larger scales, the resulting masks are upsampled

    opt is a dictionary of optional parameters. It can be used to specify 
    advanced sampling options.
    """

    # this condition may be relaxed to accomodate for non
    # standard input shapes (e.g. 299)

    #assert masks.shape[1] % (1 + 3 * levels) == 0, 'Length of the sequence must be a multiple of grid_size **2 * (1 + 3 * levels)'
    #assert input_shape[0] % grid_size == 0, "Grid size should be a multiplier of the image size."

    expanded_masks = []

    for row in range(masks.shape[0]):

        # extract the sequence
        sequence = masks[row,:]

        # convert the sequence as a set of masks, each of which will be applied
        # at the different scales
        #components = wrap_and_upscale(sequence, grid_size, input_shape, levels)
        components = split_sequence(sequence, grid_size, input_shape, levels)

        # convert the dictionnary as a mask (i.e. np.ndarray)
        out = convert_as_mask(components, input_shape, levels)

        # add if to the list
        expanded_masks.append(out)

    return expanded_masks

def reproject(sti_input, input_shape, levels):
    """
    projects the components computed at the different frequency components
    on a single space

    args
    sti_components (dict): a dictionnary with the components at each scale
    input_shape (tuple) : the size of the output image
    opt : an optional dictionnary that contains the key "size"

    components = 

    returns
    spatial_cam : the class activation map containing the coefficients at the different 
    scales

    """

    # initialize the spatial cam
    spatial_cam = np.zeros(input_shape)

    # if the input is a dictionnary, directly recover the components and the values
    if isinstance(sti_input, dict):

        for component in sti_input.keys():

            spatial_cam += cv2.resize(sti_input[component], input_shape, interpolation=cv2.INTER_CUBIC)

    # otherwise (baseline case)
    # extract the coefficients at the different scales to create the dictionnary
    # and use this dictionnary to compute the spatial cam

    elif isinstance(sti_input, np.ndarray):
        # baseline case: the input is a mask.
        # create a dictionnary 
        sizes = [int(input_shape[0] / (2 ** (levels - l))) for l in range(levels)]
        sizes.insert(0, int(input_shape[0] / 2 ** levels))
        cumsizes = np.cumsum(sizes)

        components = {}

        components['lf'] = sti_input[:cumsizes[0], :cumsizes[0]]

        components['d1'] = sti_input[cumsizes[0]:cumsizes[1], cumsizes[0]:cumsizes[1]]
        components['h1'] = sti_input[:cumsizes[0], cumsizes[0]:cumsizes[1]]
        components['v1'] = sti_input[cumsizes[0]:cumsizes[1], :cumsizes[0]]

        for i in range(1,levels):
            components['h{}'.format(i+1)] = sti_input[:cumsizes[i], cumsizes[i]:cumsizes[i+1]]
            components['v{}'.format(i+1)] = sti_input[cumsizes[i]:cumsizes[i+1], cumsizes[i]:cumsizes[i+1]]
            components['d{}'.format(i+1)] = sti_input[cumsizes[i]:cumsizes[i+1], :cumsizes[i]]

        for component in components.keys():

            spatial_cam += cv2.resize(components[component], input_shape, interpolation=cv2.INTER_CUBIC)

    return spatial_cam

def spectral_dispersion(sti):
    """
    computes the spectral dispersion of the sobol total indices

    first normalizes the indices (min max normalization)
    and then compute the entropy of the normalized vector

    the higher the entropy, the higher the spectral dispersion, as the 
    importance of the frequency components tends to be concentrated in the first
    indices (which correspond to low frequency components)

    returns: the spectral dispersion
    """

    # normalize
    norm_sti = (sti - np.min(sti)) / (np.max(sti) - np.min(sti))

    # return the entropy of the normalized sti
    return np.nanmean(sti)

def resize(image, shape):
    return cv2.resize(image, shape, interpolation=cv2.INTER_CUBIC)

def NormalizeData(data):
    """helper to normalize in [0,1] for the plots"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

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

def split_sequence(sequence, grid_size, input_shape, levels, interpolation = cv2.INTER_NEAREST):
    """
    extracts from the unidimensional sequence the coefficients corresponding
    to each level. Converts each subsequence into a mask, reshape it 
    and add it to a dictionnary where each key corresponds to 
    the location in the wavelet transform of the image
    """

    coeff_split = [(((2 ** k) * grid_size) ** 2) for k in range(levels)]
    coeff_split.insert(0, (grid_size ** 2)) # number of coeffs per level
    coeff_count = sum(coeff_split) # total number of coeffs to generate
    coeff_edge = np.sqrt(coeff_split).astype(int) # length of the mask for the different scales


    # for each sequece (i.e. line) generate masks
    # that will be applied to the cells of the wavelet transform
    thumbnails = [
        sequence[sum(coeff_split[:i]) : sum(coeff_split[:i+1])].reshape(coeff_edge[i],coeff_edge[i]) for i in range(levels+1)
    ]
    # now map each thumbnail to a part of the WT
    # and upscale if necessary
    baseline_shape = max(grid_size, int(input_shape[0] / (2 ** levels)))

    components = {
        'lf' : cv2.resize(thumbnails[0], (baseline_shape, baseline_shape), interpolation = interpolation),
        'h1' : cv2.resize(thumbnails[1], (baseline_shape, baseline_shape), interpolation = interpolation),
        'd1' : cv2.resize(thumbnails[1], (baseline_shape, baseline_shape), interpolation = interpolation),
        'v1' : cv2.resize(thumbnails[1], (baseline_shape, baseline_shape), interpolation = interpolation)
    }

    # now consider the remaining levels, from the largest to the smallest
    for i, level in enumerate(range(1,levels)):
        size = (baseline_shape * (2 ** level), baseline_shape * (2 ** level))
        components['h{}'.format(level + 1)] = cv2.resize(thumbnails[level], size, interpolation = interpolation)
        components['d{}'.format(level + 1)] = cv2.resize(thumbnails[level], size, interpolation = interpolation)
        components['v{}'.format(level + 1)] = cv2.resize(thumbnails[level], size, interpolation = interpolation)

    return components

