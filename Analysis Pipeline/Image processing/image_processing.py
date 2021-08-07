# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 10:35:10 2020

@author: Leo
"""
import copy
import numpy as np
from skimage import exposure, morphology, measure, util, filters, segmentation
from scipy.ndimage import binary_fill_holes

def process_img(img_original, mol_min_area=1300, thresh_method='otsu', smooth_method='median'):
    """ Takes the original image as input and applies various processing steps.
    
    The steps are: (for details, see resp. functions)
        1) Rescale the image to facilitate plotting
        2) Use either rescaled or original image to get a binary filter mask
        3) Update filtering mask
    
    Parameters:
    ----------
    img_original: nd-array
        the original input image
    mol_min_area: float
        the minimum area (pixels) used in morphology.remove_small_objects.
        Note: boolean input is used (the image is hence automatically labelled first)
        All objects below this area threshold are filterd out.
    thresh_method: str
        Designates thresholding method
    smooth_method: str
        Designates the filer applied after (!) thresholding.
        
    Returns:
    --------
    img_dict: dict
        {img_original: the original image
         img_rescaled: image with rescaled pixel intensities
         img_mask: first mask estimate
         img_mask_2: final mask estimate
         img_filtered: filtered image using first mask
         img_filtered_2: filtered image using updated mask
         }
    
    Note: Both masks are included to better compare final results. It is advised to
    only use the updated mask to ensure maximum efficiency output.
    """
    # Delete erroneous measurements
    img_normalized = copy.deepcopy(img_original)
    img_normalized[img_normalized < -0.5] = 0
    
    # Rescale pixel intensities for better plotting
    p_1, p_2 = np.percentile(img_normalized, (2, 98))
    img_rescaled = exposure.rescale_intensity(img_original, in_range=(p_1, p_2))
    
    # Get first estimate of filter mask
    img_mask = get_mask(img_rescaled, min_area=mol_min_area,
                        thresh_method=thresh_method, smooth_method=smooth_method)
    
    # Get filtered image
    img_filtered = copy.deepcopy(img_original)
    img_filtered[img_filtered < 0] = 0
    img_filtered[img_mask == False] = 0
    
    # Update filter mask
    img_mask_2 = update_mask(img_filtered, min_area=mol_min_area,
                             thresh_method=thresh_method, smooth_method=smooth_method)
    
    # Update filtered image
    img_filtered_2 = copy.deepcopy(img_original)
    img_filtered_2[img_filtered_2 < 0] = 0
    img_filtered_2[img_mask_2 == False] = 0
    
    # Create dictionary to look up images
    img_dict = {'img_original': img_original,
                'img_rescaled': img_rescaled,
                'img_mask': img_mask,
                'img_mask_2': img_mask_2,
                'img_filtered': img_filtered,
                'img_filtered_2': img_filtered_2
                }
    
    return img_dict


def get_molecules(img_dict, img_meta_data):
    """ Labels the filtered image to identify molecules. Then returns a list of filtered molecules. 
    Adapted from Sebastian Konrad.
    Steps:
        1) Image is labelled with connectivity n = 2.
        2) Assign bounding box to each label (i.e. molecule)
        3) Remove parts of other molecules that are in the same box
        4) Pad each molecule with a 10 pixel border
    
    Parameters:
    ----------
    img_dict: dict
        see img_processing function for keys. 
        Only uses the key 'img_mask_2', which is the updated filter mask
    img_meta_data: dict
        see import_data module for keys.
        Needed for molecule border padding.
        
        
    Returns:
    --------
    molecules: list
        molecules[i] is dict w/ keys
            mol_original: the original image depicting a single molecule
            mol_filtered: the filtered image depicitng a single molecule
            mol_bbox: the bounding box of the molecule
    """
    molecules = []
    
    # Labels BINARY input image corresponding to filtering mask
    img_filled = binary_fill_holes(img_dict['img_mask_2'])
    img_labelled = morphology.label(img_filled, connectivity=2)
    
    # Loop through all labelled regions (i.e. molecules)
    for region in measure.regionprops(img_labelled):
        curr_molecule = []
        minr, minc, maxr, maxc = region.bbox
        
        # Remove parts of other molecules that are in the same box
        mol_filtered_box = copy.deepcopy(img_dict['img_filtered_2'][minr:maxr, minc:maxc])
        mol_labelled_box = img_labelled[minr:maxr, minc:maxc]
        mol_filtered_box[mol_labelled_box != region.label] = 0

        # Pad each molecule with a 10 pixel border of the original image or 0 if at the edges of the orig image
        if np.amin([minr, minc]) < 10 or maxr > (img_meta_data['x_pixels']-10) or maxc > (img_meta_data['y_pixels']-10):
            curr_molecule.append(util.pad(img_dict['img_original'][minr:maxr, minc:maxc], pad_width=10, mode='constant'))
        else:
            curr_molecule.append(img_dict['img_original'][minr-10:maxr+10, minc-10:maxc+10])
        
        # Pad the filtered box with a 10 pixel border 
        curr_molecule.append(util.pad(mol_filtered_box, pad_width=10, mode='constant'))
        
        # Append region boundaries
        curr_molecule.append([minr, minc, maxr, maxc])
        
        molecules.append({'mol_original': curr_molecule[0],
                          'mol_filtered': curr_molecule[1],
                          'mol_bbox': curr_molecule[2]})
        
    return molecules
    

def update_mask(img_original, sigma=0.5, min_area=100,
                thresh_method='otsu', smooth_method='median'):
    """Applies a second round of filtering and threshold operations to deliver an updated filter mask.

    Parameters:
    ----------
    img_original: nd_array
        the input image
    sigma: float
        if not zero, apply a gaussian filter with sd sigma
    thresh_method: str
        a string indicating the thresholding method
    smooth_method: str
        the filter to be applied after thresholding
    
    Returns:
    -------
    mask: ndarray
        boolean array containing the segmented image

    Notes:
    ------
    For low contrast images, triangle thresholding is preferred.
    Triangle thresholding doesn't need a gaussian filter.
    The point of this mask is that sometimes, even after applying a first Median filter,
    some molecules were still having a zaggy border line. Such a zaggy border line
    prevents successful skeletonization using skimage's morphology.skeletonize(method='lee'). 
    This is also to avoid getting faulty branches during skeletonization 
    (i.e. branches that count as branchpoints)
    It is HIGHLY recommended to use Median Filter as a smoothing method. 
    """
    
    # Get second threshold
    if thresh_method == 'otsu':
        thresh = filters.threshold_otsu(img_original)
    elif thresh_method == 'triangle':
        thresh = filters.threshold_triangle(img_original)
    else:
        raise ValueError('Unknown method %s. Valid methods are otsu,'
                         ' and triangle.' % thresh_method)
    
    # Apply threshold
    mask = img_original > thresh
    
    # Apply smoothing filter
    if smooth_method == 'gaussian':
        mask = filters.gaussian(mask, sigma)
    elif smooth_method == 'median':
        mask = filters.median(mask)
    else:
        raise ValueError('Unknown method %s. Valid methods are gaussian,'
                         ' and median.' % smooth_method)
    
    # Brighten image for skeletonization
    mask = morphology.binary_closing(mask)
    
    return mask

def get_mask(img_original, sigma=0.5, thresh_method='otsu',
                   min_area=100, smooth_method='median'):
    """Applies filtering and thresholding operations to deliver a filter mask.
    Steps: 
        1) Applies gaussian filter (skimage.filter.gaussian)
        2) Thresholds image (skimage.filter.threshold_otsu)
        3) Removes small objects (skimage.morphology.remove_small_objects)
        4) Applies a median filter to avoid zaggy molecule borders
        5) Clears border (skimage.morphology.clear_border)

    Parameters:
    ----------
    img_original: nd_array
        the input image
    sigma: float
        if not zero, apply a gaussian filter with sd sigma
    thresh_method: str
        a string indicating the thresholding method
    min_area: int
        the minimum area (pixels) used in morphology.remove_small_objects.
        All objects below this area threshold are filterd out (!).
    smooth_method: str
        the filter to be applied after thresholding
    
    Returns:
    -------
    mask: ndarray
        boolean array containing the segmented image

    Notes:
    ------
    For low contrast images, triangle thresholding is preferred.
    Triangle thresholding doesn't need a gaussian filter.
    It is HIGHLY recommended to use median filtering as smoohting method.
    """
    # Apply gaussian filter
    if sigma > 0:
        img_gaussian = filters.gaussian(img_original, sigma)
        
    # Get thresholds
    if thresh_method == 'otsu':
        thresh = filters.threshold_otsu(img_gaussian)
    elif thresh_method == 'triangle':
        thresh = filters.threshold_triangle(img_gaussian)
    else:
        raise ValueError('Unknown method %s. Valid methods are otsu,'
                         ' and triangle.' % thresh_method)
    
    # Get mask
    mask = img_gaussian > thresh # apply threshold
    mask = morphology.remove_small_objects(mask > 0, min_size=min_area) # filter small strands
    
    # Apply filter to smoothen edges left over from thresholding
    if smooth_method == 'gaussian':
        mask = filters.gaussian(mask, sigma)
    elif smooth_method == 'median':
        mask = filters.median(mask)
    else:
        raise ValueError('Unknown method %s. Valid methods are gaussian,'
                         ' and median.' % smooth_method)
    
    # Brighten image for skeletonization
    mask = morphology.binary_closing(mask)
    # Remove border
    mask = segmentation.clear_border(mask)
    
    return mask
