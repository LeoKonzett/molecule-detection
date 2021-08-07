# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 18:01:24 2020

@author: Leo
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from scipy.spatial import cKDTree
from scipy.linalg import norm
from scipy.interpolate import Rbf
import interp_funcs

def height_distribution_pipeline(analyzed_linear_DNA, max_radius=5, symmetrize=False):
    """ Pipeline to calculate the average pixel height distribution around a single trace point.
        The height distribution is NOT(!) the AFM cantilever tip. The height distribution can be
        used to simulate the tip-sample dilation for a one-pixel wide DNA backbone.
    
    Parameters:
    -----------
    analyzed_linear_DNA: list of arrays
        list of linear-only DNA strands. Each linear_DNA object holds its coordinate trace.
        
    max_radius: int
        the maximum distance up to which pixels around the trace are used to calculate the
        height distribution. On average, the intensity is zero for distances greater than 4.5
        
    symmetrize: bool
        indicates whether or not to ascertain radial symmetry of the distribution. On average
        the excentricity is 1.02, thus the symmetrizing option is not needed for the images at hand.
        Can be included for images with *very* few linear strands.
    
    
    Returns:
    --------
    height_distribution: dict
        height_dist_fct: a scipy.interpolate Rbf object yielding the average pixel height
                        expected at distance d from a trace point. d is the euclidean norm of 
                        the vector connecting a pixel with its nearest neighbor in 'trace'
                        
        height_dist_hist: The normalized histogram used for RBF interpolation.
        
        rel_coords_dist: The histogram quantifying the amount of relative distance vectors that have
                        endpoints lying in a given 1x1 pixel-wide grid cell 
                        
        height_dist_excentricity: float value describing the excentricity of the distribution.
        
        height_dist_width: float value describing the radial distance at which the heights become approx. zero
        
    Adapted from Louis de Gasté, Sebastian Konrad.
    """
    # Initialize lists to calculate tip histogram and set parameters.
    rel_coords, heights = [], []
    
    # Choose number of interpolation points
    int_factor = 15
    
    # Only use linear DNA strands to estimate tip shape
    for linear_DNA in analyzed_linear_DNA:
        
        # Check if linear_DNA object already holds an optimized trace
        if 'trace_refined' in linear_DNA.mol_pars.keys():
            # If True, use optimized trace for height estimation
            trace = np.asarray(linear_DNA.mol_pars['trace_refined'])
        
        else:
            # If False, use equidistant trace for height estimation
            trace = np.asarray(linear_DNA.mol_pars['trace_equidistant'])
        
        # Interpolate trace for subsequent pixel collection
        trace_int = interp_funcs.interpolate_trace(trace, int_factor=int_factor)
        
        # Get distances from each pixel point within 'max_radius' to its nearest neighbor in 'trace'
        pixel_locs, indptr, dist_rel = get_rel_coords(trace_int, 
                                                       linear_DNA.mol_filtered.shape, max_radius)
        
        # Use filtered molecule image to collect pixel heights
        rel_coords += list(dist_rel)
        heights += list(np.asarray(
            [linear_DNA.mol_filtered[point[0], point[1]] for point in pixel_locs]))
        
    # Get distribution function and its parameters (i.e. width, excentricity)
    height_dist_function, height_dist_hist, rel_coords_dist = height_dist_estimation(
        rel_coords, heights, symmetrize)
    excentricity = height_dist_excentricity(height_dist_function)
    width = height_dist_width(height_dist_function, max_radius)
    
    # Set up dictionary
    height_distribution = {'height_dist_fct': height_dist_function,
                'height_dist_hist': height_dist_hist,
                'rel_coords_dist': rel_coords_dist,
                'height_dist_excentricity': excentricity,
                'height_dist_width': width}
    
    return height_distribution

def height_dist_estimation(rel_coords, heights, symmetrize=False):
    """ Method to calculate the average pixel height expected around a point i in 'trace'
        Algorithm works as follows:
            1) Calculate the 4 nearest-neighbor integer pairs of each distance vector in 'rel_coords'
            2) Assign weights to the 4 nn-pairs based on how close the endpoint of the distance
                vector is to each of them
            3) Each distance vector has a corresponding pixel height value
            4) Count the number of vectors that have endpoints lying in each grid cell
            5) Assign an average pixel height to each grid cell
            6) Use the established grid, and the average pixel height as input for RBF interpolation

    Parameters:
    -----------
    rel_coords:  nd_array, shape (num_points, 2)
        Contains the relative position of each pixel to the trace.
    heights: nd_array, shape (num_points, )
        Contains the height of each pixel
    symmetrize: bool
        indicates whether or not to ascertain radial symmetry the tip shape. 
        
    Returns:
    -------
    height_dist_fct: 
        The Rbf fitted function to the average pixel height histogram
    height_dist_hist: 
        The normalized pixel height distribution histogram
    rel_coords_dist: 
        The histogram quantifying the amount of relative distance vectors that have
        endpoints lying in a given 1x1 pixel-wide grid cell 
    """
    # Calculate the four closest pixel coordinate (int) neighbors for each relative coordinate point
    coords_nn_int = np.asarray([np.array([[np.floor(coord[0]), np.floor(coord[1])],
                               [np.floor(coord[0]), np.ceil(coord[1])],
                               [np.ceil(coord[0]), np.floor(coord[1])],
                               [np.ceil(coord[0]), np.ceil(coord[1])]]).astype(int) for coord in rel_coords])

    # Get the weighted nearest neighbor integer coordinates
    # Each weight is determined by how much the relative coordinate point is away from a int-tuple
    offsets = abs(rel_coords - np.ceil(rel_coords))
    weights_coords_nn = np.asarray([np.array([offset[0] * offset[1],
                                              offset[0] * (1 - offset[1]),
                                              (1 - offset[0]) * offset[1],
                                              (1 - offset[0]) * (1 - offset[1])]) for offset in offsets])
    
    # Stack the AFM heights 4 times since there are 4 neighboring int-tuples for each relative distance
    heights = np.vstack((heights, heights, heights, heights)).T
    
    # Flatten / reshape the arrays for histogram calculation
    heights, weights_coords_nn = heights.flatten(), weights_coords_nn.flatten()
    coords_nn_int = coords_nn_int.reshape([-1, 2])

    # Determine the maximal and minimal relative coordinate point
    # Those two points span the 2-D grid used to estimate the average pixel height distribution
    max_rel_coords = np.max(coords_nn_int, axis=0)
    min_rel_coords = np.min(coords_nn_int, axis=0)
    
    # Get the grid described above: +2 to ensure that the outermost points ->
    # -> are falling into a histogram bin
    bins = [np.arange(low, high + 2) for low, high in zip(min_rel_coords, max_rel_coords)]
    
    # Calculate the combined height of all pixels that have have approx. equal relative distance vectors
    height_dist = np.histogramdd(coords_nn_int, 
                               weights=heights * weights_coords_nn, bins=bins)[0]
    
    # Calculate the distribution of relative distance vectors in the 2D grid described above
    rel_coords_dist = np.histogramdd(coords_nn_int, 
                                   weights=weights_coords_nn, bins=bins)[0]
    
    # Normalize the combined height values
    height_dist_norm = np.divide(height_dist, rel_coords_dist, 
                               out=np.zeros_like(height_dist), where=rel_coords_dist!=0)

    # Set up the grid for RBF interpolation
    # +1 to ensure that the upper limit is included in np.arange
    bins = [np.arange(low, high + 1) for low, high in zip(min_rel_coords, max_rel_coords)]
    mesh_coords = np.array(np.meshgrid(*bins)).T
    
    # Convert the normalized histogram to an actual function by RBF interpolation
    if symmetrize == False:
        height_dist_fct = Rbf(mesh_coords[:,:,0], 
                             mesh_coords[:,:,1], height_dist_norm, function='linear')
    else:
        height_dist_fct = symmetrize_tip_shape(mesh_coords, height_dist_norm, rel_coords_dist)
        
    return height_dist_fct, height_dist_norm, rel_coords_dist

def get_rel_coords(trace, img_shape, dist_limit=5.0, exc=1, clip_endpoints=True):
    """ Method to calculate the distance of a pixel to its nearest-neighbor in 'trace'. 
    The distance is returned as vector whose coordinates denote the displacement of the
    pixel to its nearest neighbor in 'trace'. 
    
    Parameters:
    ----------
    trace: array of float64
        interpolated one-pixel wide DNA backbone of shape (num_points, 2)
    img_shape: tuple
        the shape of the input image
    dist_limit: float
        the maximum distance up to which pixel points are taken into account
    excentricity: int (std x-axis / std y-axis)
        scaling factor for the vertical axis of the grid. An excentricity > 1 compresses 
        the grid vertically. 
        
    Note:
    ------
    The excentricity is almost 1 (1.02 on average). Thus, the effect on the overall calculation
    is negligible. 
    Note that multiple pixels can have the same nearest-neighbor in trace. This effect is especially
    prominent at sharp bends and must be addressed in the optimization algorithm.
    
    Returns:
    -------
    pixel_locs: array of int32
        array holding all pixel coordinate tuples which are within 'dist_limit' of 'trace'
    indptr: tuple of ints
        indices of the nearest neighbor point pair in 'trace' for each point in pixel_locs
    dist_rel: array of float64, shape: (num_points, 2)
        the coordinate-wise distance between a pixel point and its nn in trace
    """
    # Initialize distortion vector
    distortion = [1, exc] 
    
    # Set up 2D grid with row number (x_max - x_min) and column number (y_max - y_min)
    x_min, y_min = np.floor(np.min(trace, axis=0) # set up grid
                            - dist_limit).astype(int).clip(min=0)
    x_max, y_max = np.ceil(np.max(trace, axis=0)
                           + dist_limit).astype(int).clip(max=img_shape)

    pixel_locs = np.mgrid[x_min: x_max, y_min: y_max] # flesh out grid
    pixel_locs = np.reshape(pixel_locs, (2, -1)).T # flatten grid
    
    # Initialize tree for nn-search
    tree = cKDTree(trace / distortion)
    dist_to_nn, indptr = tree.query(pixel_locs / distortion,  # query tree
                                    distance_upper_bound=dist_limit)
    
    pixel_locs = pixel_locs[dist_to_nn != np.inf] # Remove pixels that are not within dist_limit
    indptr = indptr[dist_to_nn != np.inf] # Repeat for index list
    dist_rel = pixel_locs - trace[indptr] # Get relative distance vectors
    
    # If desired, all pixels that have the first or last trace point as a nearest neighbor are discarded
    if clip_endpoints:
        mask = (indptr != 0) & (indptr != len(trace) - 1) # remove end-points
        pixel_locs, indptr, dist_rel = pixel_locs[mask], indptr[mask], dist_rel[mask]
    
    return pixel_locs, indptr, dist_rel

def height_dist_excentricity(height_dist_fct):
    """ Calculates the excentricity of the pixel height distribution function to check whether it
    can be interpolated by RBF's.

    Parameters:
    ----------
    height_dist_fct: scipy.interpolate Rbf object
        function that estimates the average pixel height in the vicinity of a trace point.
        
    Returns:
    -------
    excentricity: float
        excentricity of the distribution
    
    Note:
    -----
    excentricity * std_y_axis = std_x_axis
    """
    steps = 101
    linspace = np.linspace(-10, 10, steps)
    x_shape = height_dist_fct([0]*steps, linspace) # Height distribution along the x axis
    y_shape = height_dist_fct(linspace, [0]*steps) # Height distribution along the y axis
    x_shape -= np.amin(x_shape)
    y_shape -= np.amin(y_shape)
    std1 = np.sqrt((x_shape * linspace ** 2).mean())
    std2 = np.sqrt((y_shape * linspace ** 2).mean())

    excentricity = std1 / std2

    return excentricity


def symmetrize_tip_shape(mesh_coords, height_dist_norm, rel_coords_dist):
    """ Method to ascertain radial symmetry of the pixel height distribution function
    
    Parameters:
    ----------
    mesh_coords: np.meshgrid(*bins) - array
        a 2-D meshgrid symbolizing the relative distance coordinate array
    height_dist_norm: nd-array
        histogram with its values normalized by the sample count
    rel_coords_dist: nd-array, same shape as tip_shape_norm
        sample-count histogram
    
    Returns:
    -------
    tip_shape_fct: scipy.interpolate Rbf object
        the pixel height distribution obtained by RBF interpolation fit
    
    Note:
    ------
    Not needed for the images utilized.
    """
    # Get distance from grid points to center of grid (center of tip shape)
    dist_to_centre = (np.linalg.norm(mesh_coords, axis=-1)).round().astype(int)
    
    # Note: For each grid point that has the same (approx.) distance to the grid centre,
    # we add the weights (i.e. the heights * sample-counts) together. 
    weights_per_dist = np.bincount(dist_to_centre.flatten(),
                     weights=height_dist_norm.flatten() * rel_coords_dist.flatten())
    samples_per_dist = np.bincount(dist_to_centre.flatten(),
                    weights=rel_coords_dist.flatten())
    weights_norm = np.divide(weights_per_dist, samples_per_dist, 
                                    out=np.zeros_like(weights_per_dist), where=samples_per_dist!=0)
    
    # return the weights to a radially symmetrical grid
    symmetrized_weights = weights_norm[dist_to_centre]
    
    return Rbf(mesh_coords[:,:,0], mesh_coords[:,:,1], symmetrized_weights, function='linear')


def height_dist_width(height_dist_fct, max_radius, drop_off=0.25):
    """ Method to determine the distance above which the pixel heights have less than
    25% of the height at the center (i.e., the trace point)
    
    Parameters:
    ----------
    height_dist_fct: scipy.interpolate.Rbf object
        the pixel height distribution function
    global_max_width: 
    """
    
    # Get center height
    centre_height = height_dist_fct(0,0)
    
    # Sample an "Einheitskreis" with radius = 1
    angles = np.linspace(0, 2*np.pi, num=16, endpoint=False)
    sample_points = np.array([np.cos(angles), np.sin(angles)])
    
    # Get maximum distance from interpolated RBF's
    dist_max = max(np.linalg.norm(height_dist_fct.xi, axis=0))
    
    # Loop through different radii, up to a max. radius
    for radius in np.linspace(0, dist_max, num=40):
        
        # Get mean, max pixel intensity for a given radius
        max_intensity = np.max(height_dist_fct(*(sample_points * radius)))
        
        # Define drop-off as 10% of original intensity
        if max_intensity < (centre_height * drop_off):
            return radius

    return max_radius

#%% Outdated
def plot_tip_shape(tip_dict, grid_range=5):

    # Set up quadratic grid
    num_points = 1 + 20 * grid_range
    linspace = np.linspace(-grid_range, grid_range, num_points)
    X, Y = np.meshgrid(linspace, linspace)

    # Plug the values into the tip shape function
    Z = tip_dict['tip_shape_fct'](X, Y)
    
    ################
    # Set up 3D plot
    ################
    # Three times as wide as tall
    fig = plt.figure(figsize=(10, 10), dpi=400)
    fig.suptitle('Estimated AFM tip shape')
    
    # First subplot
    ax = fig.add_subplot(2, 1, 1, projection='3d')
    #surf = ax.contour3D(X, Y, Z, 50, cmap='binary')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='binary', edgecolor='none')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title('Surface plot of estimated AFM tip shape. Tip origin at (0, 0).')
    ax.set_ylabel('Distance from center [nm]')
    ax.set_zlabel()
    ax.view_init(50, 35)
    
    # Second subplot
    ax = fig.add_subplot(2, 1, 2)
    ax.imshow(Z)
    ax.set_title('Intensity plot of the AFM tip shape')
    
    plt.show()
    
    return None
