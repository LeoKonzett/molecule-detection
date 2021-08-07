# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:56:29 2020

@author: Leo
"""
from sklearn.neighbors import NearestNeighbors
import copy
import networkx as nx
from scipy.ndimage import binary_fill_holes, convolve
from scipy.spatial import cKDTree
from scipy.linalg import norm
from skimage.morphology import skeletonize, medial_axis, label
import numpy as np



def get_skel_pars(mol_skeleton):
    """ calculates the properties of a given skeletonized molecule.
    
    Attributes:
    ----------
    mol_skeleton: nd-array
        the skeleton to analyze
    
    Returns:
    --------
    skel_dict: dict
        {skel_eps_number: int
             number of endpoints
        skel_bps_number: int
            number of branchpoints
        skel_pixel_number: int
            number of total pixels comprising the skeleton
        }
    """
    # Create copy
    mol_skel = copy.deepcopy(mol_skeleton)

    # Calculate the number of endpoints and branchpoints
    pixels_number = np.sum(mol_skel)
    skel_bp, skel_ep = get_junctions(mol_skel)
    eps_number = np.sum(skel_ep)
    bps_number = np.sum(skel_bp)
        
    # Get dictionary
    skel_dict = {'skel_eps_number': eps_number,
                 'skel_bps_number': bps_number,
                 'skel_pixels_number': pixels_number}

    return skel_dict

def order_teardrop_trace(td_indices, strep_ind):
    """ orders a teardrop trace by identifying the streptavidin molecule.
    
    Parameters:
    ----------
    mask: nd_array
        float image used to identify the streptavidin
    grain: boolean array
        the current binary strand before skeletonization
    contour: boolean array
        the contour of the current grain
    
    Returns:
    -------
    td_indices: int array
        array of shape (Num_Indices, 2) containing the contour coordinates.
    """
    td_indices = np.append([strep_ind], td_indices, axis=0) # append strep
    td_indices = order_trace(td_indices, 0) # start ordering with strep
    td_indices = np.append(td_indices, [strep_ind], axis=0) # close loop
    return td_indices

def prune_skeleton(skel, skel_ep, prune_length=0.15):
    """ method to remove faulty short skeleton branches.
    
    Parameters:
    -----------
    skel: nd_array
        the single strand skeleton image
    skel_ep: nd_array
        the endpoints of the strand skeleton
    prune_length: float
        the percentage after which the pruning iteration stops
        
    Returns:
    -------
    pruned_skel: boolean array
        same array as skel, but faulty pixels set to False
    """
    pruned_skel = np.copy(skel)
    prune_indices = np.transpose(np.nonzero(skel)).tolist()
    
    # Set pruning length
    length_of_trace = len(prune_indices)
    max_branch_length = int(length_of_trace * prune_length) # short branch limit
    
    # Identify all end points of all branches
    branch_ends = np.transpose(np.nonzero(skel_ep)).tolist()
    
    # Check for branch - and if it is delete it
    for x_b, y_b in branch_ends:
        branch_coordinates = [[x_b,y_b]]
        branch_continues = True
        temp_coordinates = prune_indices[:]
        temp_coordinates.pop(temp_coordinates.index([x_b,y_b])) # remove end point

        while branch_continues:
            tree = cKDTree(temp_coordinates)
            query_point = [x_b, y_b]
            no_of_neighbours = len(tree.query_ball_point(query_point, r=1.5))
            
            # If branch continues
            if no_of_neighbours == 1:
                _, ind = tree.query([x_b, y_b], distance_upper_bound=1.5)
                x_b, y_b = tree.data[ind].astype(int) # move one pixel
                branch_coordinates.append([x_b,y_b])
                temp_coordinates.pop(temp_coordinates.index([x_b,y_b]))

            # If the branch reaches the edge of the main trace
            elif no_of_neighbours > 1:
                branch_coordinates.pop(branch_coordinates.index([x_b,y_b]))
                branch_continues = False
                is_branch = True
            # Weird case that happens sometimes
            elif no_of_neighbours == 0:
                is_branch = True
                branch_continues = False

            if len(branch_coordinates) > max_branch_length:
                branch_continues = False
                is_branch = False

        if is_branch:
            for x, y in branch_coordinates:
                pruned_skel[x,y] = False
    return pruned_skel

def get_junctions(skel):
    """ method to identify branch and endoints of a skeleton using a conv. kernel.
    Returns two boolean filter masks.
    
    Parameters:
    ----------
    skel: nd-array
        the skeleton to analyze
    
    Returns:
    -------
    skel_bp: bool. array
        True value signals a branch point at that pixel
    skel_ep: bool. array
        True value signals an end point at that pixel
    """
    ndim = skel.ndim
    degree_kernel = np.ones((3,) * ndim) # get conv. kernel
    degree_kernel[(1,) * ndim] = 0  # remove centre pixel
    degree_image = convolve(skel.astype(int), degree_kernel,
                            mode='constant') * skel
    
    skel_bp = (degree_image > 2) # more than two neighboring pixels
    skel_ep = (degree_image == 1) # one neighbouring pixel
    return skel_bp, skel_ep

def order_trace(indices, start_node=None):
    """ method to sort a skeletonized molecule. If start_node is not None,
    the sorting algorithm uses start_node as the point to the sorting process. Else,
    the algorithm iterates over all point indices. 
    
    Parameters:
    ----------
    indices: ndarray
        the trace points of the skeletonized molecule
    start_node: int
        node of the graph where the depth-first-search starts
    num_neighbors: int
        specify how many neighbors to find
        
    Returns:
    -------
    opt_order: ndarray
        the skeleton trace points in sorted order
    """
    # Fit the estimator of the Nearest Neighbor class    
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(indices)
    
    # Get adjacency matrix, convert to graph
    adj_matrix = neigh.kneighbors_graph() 
    graph = nx.from_scipy_sparse_matrix(adj_matrix)
    
    if start_node is None:
        num_indices = len(indices) # how many coordinates
        min_dist = np.inf # cost of specific order
        opt_order = np.zeros_like(indices) # initialize optimum order
        
        # Try out every node order and return cheapest
        for node in range(num_indices):
            ordered_nodes = list(nx.dfs_preorder_nodes(graph, node))
            ordered_indices = indices[ordered_nodes]
            order_norm = ordered_indices[:-1] - ordered_indices[1:] # order cost
            order_norm = ((order_norm**2).sum(1)).sum()
            
            if order_norm < min_dist:
                min_dist = order_norm
                opt_order = ordered_indices
                
        return opt_order
    
    # Return ordered index list starting from specified node
    return indices[list(nx.dfs_preorder_nodes(graph, start_node))]

def in_circle(image, radius, origin=(0, 0), target_val=0):
        """checks whether or not a point lies within a circle with given origin"""
        
        # Loop through image
        for index, val in np.ndenumerate(image):
            # Get distance using euclidean norm
            dist_to_origin = np.sqrt((index[0] - origin[0]) ** 2 + (index[1] - origin[1])**2)
            if dist_to_origin <= radius:
                # Set pixel value to zero if within circle (i.e. strep)
                val *= target_val
        return image
    
#%% Outdated
def classify_strands(image, gauss_mask, expected_length, 
                     expected_width, prune_skeletons=True, extend_linear=False):
    """ classifies linear strands and teardrops and sorts them into lists. 
    The classification is based on the following criteria: 
        1) correct length for linear and teardrop strands
        2) correct width as a first distinction criteria by using a medial axis skeletonization
        3) correct number of endpoints and branchpoints
    
    Parameters:
    -----------
    image: ndarray
        the segmented binary input image
    gauss_mask: ndarray
        input image with gaussian filter to identify streptavidin
    expected_length: float
        the expected length of a DNA strand
        Note: this expected length should be calculated by noting that 
        1 pixel ~ 1.5 nm -> 500 bp -> 115 pixels 
    expected_width: float
        used to differentiate between teardrops and linear strands
    prune_skeletons: bool
        indictates whether or not to prune skeleton branches
    
    Returns:
    -------
    linear_backbones: list of nd_arrays
        contains the coordinates of all linear traces
    teardrop_backbones: list of nd_arrays
        contains the coordinates of all teardrop traces
    """
    linear_backbones = [] # initialize skeleton list
    teardrop_backbones = []
    
    class_statistics = {'mean_length': 0, 'mean_width': 0, 'mean_surface': 0}
    
    """# calculate DNA surface (rectangular)
    expected_surface = expected_length * expected_width
    surface_upper_bound = expected_surface * 1.5 # perhaps adjustment needed
    surface_lower_bound = expected_surface * 0.5"""
    
    # get length boundaries used for classification
    length_upper_bound = expected_length * 1.4 # maybe rewrite as interval
    length_lower_bound = expected_length * 0.6
    
    filled_img = binary_fill_holes(image) # fill image for labeling 
    grain_labels = label(filled_img) # label the image
    
    # skeletonize the image
    _, distance_to_background = medial_axis(filled_img, return_distance=True)
    skeleton_img = skeletonize(image, method='lee').astype(bool)

    """# get labeled grain contours -> used for contours by Louis
    filled_img = erosion(filled_img) # second erosion widens the contours
    grain_contours = (erosion(filled_img) ^ filled_img) * grain_labels"""

    # loop through grains
    num_grains = np.max(grain_labels)
    class_statistics['number of grains:'] = num_grains
    
    # get ints for statistics
    big_surface = 0
    small_surface = 0
    branchpoints_false = 0
    endpoints_false = 0
    false_linear_length = 0
    mean_surface, mean_length = [], []
    
    for i in range(1, num_grains + 1):
        print("current grain:", i)
        
        # Note: this iteration works by starting with true booleans. 
        # Then, various conditions are being checked. Those conditions are ordered
        # such that non-memory-intensive calculations come first.
        # if a condition is not fulfilled, the next iteration of the loop is started
        
        is_linear = True
        is_teardrop = True

        grain = (grain_labels == i) # get grain
        skel = skeleton_img * grain # get skeleton
        
        # get grain width by using medial axis skeletonization
        grain_width = np.max(distance_to_background * grain)
        
        # a width distinction is fine for a first estimate. However, small teardrops
        # could be classified as linear strands -> count endpoints
        # also, binary_fill_holes could lead to mistakes -> count endpoints
        if grain_width > expected_width: # too broad for linear strand
            is_linear = False
        if grain_width < expected_width: # too small for teardrop
            is_teardrop = False
        
        # get number of pixels of the one-pixel skeletonization algorithm by Lee
        num_pixels_per_skel = np.sum(skel)
        print('surface:', num_pixels_per_skel)
        mean_surface += [num_pixels_per_skel]
        
        # check for correct surface
        # Note: a teardrop should have the same length as a linear strand.
        # This is due to the fact that teardrops are being formed by bended
        # linear strands.
        if num_pixels_per_skel > length_upper_bound:
            big_surface += 1
            continue
        if num_pixels_per_skel < length_lower_bound:
            small_surface += 1
            continue
        
        # after a first split w.r.t width, now check endpoints
        skel_bp, skel_ep = get_junctions(skel)
        num_grain_ep = np.sum(skel_ep)
        num_grain_bp = np.sum(skel_bp)
        print('num_ep:', num_grain_ep, 'num_bp', num_grain_bp)
        
        # check whether to prune the skeleton
        # pruning serves to delete faulty branchpoints resulting from skeletonize(method='lee')
        if prune_skeletons:
            print('prune skeleton')
            skel = prune_skeleton(skel, skel_ep) # update skeleton
            skel_bp, skel_ep = get_junctions(skel) # update endpoints and branchpoints
            num_grain_ep = np.sum(skel_ep)
            num_grain_bp = np.sum(skel_bp)
            print('num_ep after pruning:', num_grain_ep, 'num_bp after pruning', num_grain_bp)

        # discard strands with branchpoints (after possibly pruning the skeleton)
        # an option to keep strands with two endpoints and up to two branchpoints 
        # is implemented since the pruning algorithm successfully removes little branches
        # but still classifies the branchpoint as such (there might be one pixel leftover)
        # Future: one could implement a uniquify junctions algorithm to avoid that...
        if num_grain_bp != 0:
            if num_grain_ep == 2 and num_grain_bp <= 2 and extend_linear:
                pass
            else:
                branchpoints_false += 1
                continue
        
        # discard strands with too many endpoints
        # this serves two clean up possible misclassifications from the width distinction
        if is_linear:
            if num_grain_ep == 2:
                 skel_indices = np.transpose(np.nonzero(skel)) # get skeleton coordinates
                 trace = order_trace(skel_indices) # order the trace
                 length = sum(norm(trace[1:] - trace[:-1], axis=1)) # get its length, is this a double check
                 mean_length += [length]
                 print('length:', length)
                 if (length_lower_bound < length < length_upper_bound):
                     print('classified: linear')
                     linear_backbones.append((trace))
                 else:
                    false_linear_length += 1
                    continue
                
            else:
                endpoints_false += 1
                continue
            
        if is_teardrop: 
            if num_grain_ep == 0:
                 skel_indices = np.transpose(np.nonzero(skel)) # get skeleton coordinates
                 trace = order_teardrop_trace(gauss_mask, grain, skel_indices)
                 print('classified: teardrop')
                 teardrop_backbones.append((trace))
                
            else:
                endpoints_false += 1
                continue
        
    print(np.asarray(mean_length).mean(), np.asarray(mean_surface).mean())
    print(" bps false:", branchpoints_false, "eps false:", endpoints_false,
          "surface too big:", big_surface, "surface too small:", small_surface, 
          'false lin. length:', false_linear_length)
    
    return linear_backbones, teardrop_backbones

def order_teardrop_trace_old(mask, grain, td_indices):
    """ orders a teardrop trace by identifying the streptavidin molecule.
    
    Parameters:
    ----------
    mask: nd_array
        float image used to identify the streptavidin
    grain: boolean array
        the current binary strand before skeletonization
    contour: boolean array
        the contour of the current grain
    
    Returns:
    -------
    contour_indices: int array
        array of shape (Num_Indices, 2) containing the contour coordinates.
    """
    shape = mask.shape
    strep_ind = np.argmax(mask * grain) # get index
    strep_ind = np.unravel_index(strep_ind, shape) # format index
    
    td_indices = np.append([strep_ind], td_indices, axis=0) # append strep
    td_indices = order_trace(td_indices, 0) # start ordering with strep
    td_indices = np.append(td_indices, [strep_ind], axis=0) # close loop
    return td_indices

 