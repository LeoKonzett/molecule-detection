# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:02:28 2020

@author: Leo
"""
import numpy as np
from numpy.linalg import norm
import copy
import interp_funcs
import height_estimation_pipeline as hep

def refine_trace(image, trace, height_dist, improv_params=None, mol='teardrop'):
    """
    A method to optimize the trace point pairs obtained by the skeletonization algorithm.
    The dict improv_params holds all the parameters necessary for tweaking the algorithm.
    
    Parameters:
    -----------
    image: ndarray
        iput image holding the pixel intensity values (corresponding to the AFM heights)
    trace: ndarray
        the trace to optimize
    height_dist: dict
        dictionary holding the pixel height distribution function
    mol: str
        the type of molecule to optimize
    improv_params: dict
        see .main() file for key description
        
    Returns:
    -------
    trace: ndarray
        the optimized trace
    """
    trace = copy.deepcopy(trace) # copy initial trace
    
    if improv_params is not None:
        # Perform optimization for _ iterations
        for _ in range(improv_params['num_iter']):
            
            # NOT ADVISED; JUST FOR COMPARISON WITH LOUIS!!
            if improv_params['allow_segment_breakup']:
                trace = adjust_lengths(trace, grad_params=improv_params['grad_params'])
            
            # Interpolate trace
            trace_int = interp_funcs.interpolate_trace(trace, improv_params['steps_per_pixel'])
            
            # Get all pixels whose distance to their nn in 'trace_int' is < ref_radius
            pixel_locs, indptr, rel_coords = hep.get_rel_coords(trace_int, image.shape,
                                            dist_limit=improv_params['ref_radius'], 
                                            exc=height_dist['height_dist_excentricity'],
                                            clip_endpoints=False)
            
            # Compare pixel heights expected by height distribution function ->
            # -> with the pixel heights obtained by AFM
            height_diff_map = height_dist['height_dist_fct'](*(rel_coords.T)) - image[tuple(pixel_locs.T)]
            
            # Safety check in case 'ref_radius' > 'height_dist_width'
            # Set erroneous offsets to average value
            height_diff_map[height_diff_map > 0.35] = 0.04
            
            # Get mean squared error
            height_diff_mse = np.mean((height_diff_map ** 2))
            
            # Calculate gradients which are used to optimize the trace
            gradients = get_gradients(trace, trace_int, indptr, rel_coords,
                                      height_diff_map, improv_params)
            
            # If teardrop, set the displacement of trace points with index in ['points_to_fix] ->
            # -> to zero. Not used anymore, but maybe helpful for further application
            if mol == 'teardrop':
                gradients[improv_params['points_to_fix']] *= 0
            
            # Set max displacement for each point in trace
            if improv_params['gradient_thresholds'] is not None:
                gradients = limit_grd_displacement(
                    gradients, improv_params['gradient_thresholds'])
            
            # Store each interation step to analyze convergence
            if 'analyze_data' in improv_params.keys():
                analyze_data = improv_params['analyze_data']
                
                # DNA backbone change at each iteration
                if 'trace' in analyze_data.keys():
                    analyze_data['trace'] += list([copy.deepcopy(trace)])
                    
                # Gradient change at each iteration
                if 'gradients' in analyze_data.keys():
                    analyze_data['gradients'] += list([gradients])
                    
                # MSE for each iteration
                if 'offsets_mse' in analyze_data.keys():
                    analyze_data['offsets_mse'] += list([height_diff_mse])
                    
            # Update trace
            trace += gradients * improv_params['learning_rate']

    return trace

def get_gradients(trace, trace_int, indptr, rel_coords, height_diff_map,
                  improv_params=None):
    """
    A method to calculate the increments (i.e. gradients) which are used to update
    (i.e. move) the trace points. Example: point_final = point_original + const * gradient(point_original).
    The steps of the algorithm are as follows:
        1) Add the height offset of all pixels that have the same nn in 'trace_int' together
        2) Move the corresponding trace point in the direction indicated by the offets
        3) Add regularization parameter to ensure equidistance
        4) Add regularization parameter to avoid overfitting
    
    Parameters:
    ----------
    trace: ndarray w/ shape: (num_points, 2)
        the coordinate trace
    trace_int: ndarray, shape (num_points * interp_factor, 2)
        the interpolated coordinate trace
    indptr: ndarray w/ shape: (num_points,)
        relates each point in pixel_locs to its nearest neighbor in 'trace'
        example: indptr = [0, 2, 1] indicates that the nearest neighbor of
        pixel_locs[2] is trace[1]
    height_diff_map: ndarray w/ shape: (num_points,)
        1-D array holding the height differences, same shape as indptr
    rel_coords: nd_array w/ shape: (pixel_locs,)
        float value denoting the distance vector from a point in pixel_locs to its 
        nn in 'trace'
    
    Returns:
    --------
    gradients: ndarray w/ same shape as trace
        the increments used for displacement of the trace points
        
    """
    # Initialize gradients
    gradients_final = np.zeros_like(trace)
    grad_params = improv_params['grad_params']
    
    # Get vector directions (starting from origin (0, 0))
    normed_coords = (rel_coords.T / (norm(rel_coords, axis=1) + 1e-9)).T
    
    ########
    # Essential Part:
    ########
    
    # Calculation as done by Louis
    if not grad_params['new_gradients_fine']:
        gradients_fine = np.vstack(
            (-np.bincount(indptr, weights=height_diff_map * normed_coords[:,0], minlength=len(trace_int)),
             -np.bincount(indptr, weights=height_diff_map * normed_coords[:,1], minlength=len(trace_int)))).T
    
    # My calculation
    elif grad_params['new_gradients_fine']:
        # Calculate the ratio of nearest-neighbor points with positive relative coordinates ->
        # -> with respect to nearest-neighbor points with negative relative coordinates
        skew_x = np.absolute(np.bincount(
            indptr, weights=normed_coords[:,0], minlength=len(trace_int)))
        
        # Set skew values smaller than 1 to 1
        skew_x[np.abs(skew_x) <= 1] = 1.
        
        # Add the offsets of those pixels together which have the same nearest-neighbor trace point
        # Minus sign indicates that we move in the *opposite* direction according to grad. descent theory
        # Offsets of points counteract the offsets of points opposite to the trace as intended
        offset_combined_x = -np.bincount(
            indptr, weights=height_diff_map * normed_coords[:,0], minlength=len(trace_int))
        
        # Normalize the combined offsets by their respective skew ratio if the absolute value ->
        # -> exceeds 1. This indicates a comparatively large skew, and reduces the offset value.
        norm_quantity_x = np.divide(offset_combined_x, skew_x, 
                                    out=np.zeros_like(offset_combined_x))
    
        # Repeat the exact same steps for y-axis
        skew_y = np.absolute(np.bincount(
            indptr, weights=normed_coords[:,1], minlength=len(trace_int)))
        
        skew_y[np.abs(skew_y) <= 1] = 1.
    
        offset_combined_y = -np.bincount(
            indptr, weights=height_diff_map * normed_coords[:,1], minlength=len(trace_int))
        
        norm_quantity_y = np.divide(offset_combined_y, skew_y, 
                                    out=np.zeros_like(offset_combined_y))
        
        # Set unregularized gradients of the interpolated trace
        gradients_fine = np.vstack((norm_quantity_x, norm_quantity_y)).T
    
    ########
    # Note for Leo:
    # Outdated !!
    # All grid_points that have the same nn (same index) in trace get summed and are put in one bin.
    # Note: The minimum length is the number of elements in fine trace. 
    # The maximum length of gradients_fine is the number of elements in indptr -
    #  - this would be the case if no point in grid_points has the same nearest neighbor -
    # - in trace as any other point in grid_points.
    # Note: first point in gradients_fine corresponds to all points whose nn is the first point in fine trace.
    # Goal: instead of updating each point in the fine trace, we want 
    # to update only those points which constitute the start / end of a segment of the WLC.
    # those start / end points correspond to the original trace
    ##########
    
    # Initialize gradients for the original (i.e., not interpolated) trace
    gradients_coarse = np.zeros_like(trace)
    
    # Note: If len(gradients_fine) is len(fine_trace), the mult_factor is ->
    # -> just the interpolation factor used to interpolate the trace.
    # Can be used as sanity check.
    mult_factor = int((len(gradients_fine) - 1) / (len(trace) - 1))

    # Set scale to distribute the gradients of the interpolated trace to the coarse trace
    mult_scale = np.linspace(1, 0, mult_factor, endpoint=False)

    ##############
    # Note for Leo, and to explain a bit more whats going on:
    # let's assume that we have a gradient for each point in fine_trace. 
    # how would we distribute those gradients to the coarse trace? 
    # an idea would be to split the fine gradients into packs of 10 (or the mult factor)
    # each, and add up the value in those packs with some weights.
    # reshape: the split below is the easiest since it starts with the first point and adds the next
    # 10 points to the first one. 
    # moveaxis: transform (n, 2) to (2, n) for mat. mult.
    # 1: add points from 0 to 1 (excl. 1), from 1 to 2, and so on...
    # np.matmul does just that: it treats a nd array with n > 2 as a stack of matrices
    # so shape (N, 10, 2) means that we get a matrix mult. N times
    # 2: add points from 1 to 2 (1, 2 excl.)
    ##############
    
    # Multiply each gradient of the interpolated trace with its corresponding scaling factor, and ->
    # -> add the interpolated gradients together in batches of size *mult_factor* 
    # Addition starts with first (index: 0) interpolated gradient vector, meaning that ->
    # -> the interpolated gradients to the *right* of each coarse trace point are added.
    # Example for mult_factor = 10: gradients_coarse[0] = gradients_fine[0] * 1 + 
    # + gradients_fine[1] * 0.9 + ... + gradients_fine[9] * 0.1
    gradients_coarse[:-1] += np.matmul(np.moveaxis(
        gradients_fine[:-1].reshape((-1, mult_factor, 2)) , 1, 2), mult_scale)
    
    # Repeat the step described above, but start addition from the second (index: 1) ->
    # -> and reverse the scaling factor. 
    # This means that the interpolated gradients to the *left* of each coarse trace ->
    # -> point are added together.
    # Example for mult_factor = 10: gradients_coarse[1] = gradients_fine[1] * 0.1 + 
    # + gradients_fine[2] + 0.2 + ... + gradients_fine[9] + 0.9 + gradients_fine[10] * 0
    mult_scale[0] = 0 
    gradients_coarse[1:] += np.matmul(np.moveaxis(
        gradients_fine[1:].reshape((-1, mult_factor, 2)), 1, 2), np.flip(mult_scale, 0))
    
    # The last fine increment wasn't counted yet, since mult_scale[0] = mult_scale[last_index] = 0
    # Irrelevant for the overall calculation, but included for sake of completeness
    gradients_coarse[-1] += gradients_fine[-1]
    
    # Set final gradients before regularization
    gradients_final += gradients_coarse
 
    # Include regularization parameters
    if grad_params is not None:
        if grad_params['set_equidistant']:
            gradients_final += set_equidistant(
                trace, grad_params=grad_params, seg_length=improv_params['segment_length'])
        if grad_params['limit_overfit']:
            gradients_final += limit_overfit(trace, grad_params=grad_params)
            
    return gradients_final


def set_equidistant(trace, grad_params, seg_length):
    """ Method to reduce movement of trace points parallel to the line connecting them.
    
    Parameters:
    ----------
    trace: nd-array
        the 'trace' object used to define the direction of displacement
    grad_params: dict
        the dictionary holding the cost penalty assigned to the displacement
    seg_length: float
        the length which the segments spanning the trace coordinate points should have
    
    Returns:
    -------
        regularization: nd-array
            the increment added to the initially calculated gradients to ensure equidistance
            
    Notes Leo:
     high offsets should be penalized as outlined in the reg. docs
     trace points are only allowed to move along the line connecting them
     Idea for future: Trace points should not move along their connecting line at all!
     Question already on stackoverflow for this interesting problem, proceed once thesis finished
    """
    # Initialize regularization increments
    regularization = np.zeros_like(trace)
    
    # Get bond vectors
    # Note: direction vectors point from p1 to p2, i.e. from p_n to p_{n+1}
    # Expected norm should be equal to chosen segment length, and is exactly the ->
    # -> segment length for the first iteration
    trace_vectors = np.diff(trace, axis=0)
    trace_norms = np.linalg.norm(trace_vectors, axis=1)
    
    # Calculation by Leo
    if grad_params['lr_updated']:
        # Calculate direction along which the trace points can move
        dir_lp = trace[-1] - trace[-2]
        directions = trace[2:] - trace[:-2]
        
        # Get normalized direction
        normed_dir_lp = dir_lp / (np.linalg.norm(dir_lp) + 1e-9)
        normed_dirs = directions / (np.linalg.norm(directions, axis=1)[:, np.newaxis] + 1e-9)
        
        # Get regularization penalties
        offset_lp = trace_norms[-1] - seg_length
        offset_gen = trace_norms[:-1] - seg_length
        
        cost_lp = (normed_dir_lp.T * offset_lp).T * grad_params['beta']
        cost_gen = (normed_dirs.T * offset_gen).T * grad_params['beta']
        
        # Set up regularization
        #regularization[0] -= cost_fp
        regularization[-1] -= cost_lp
        regularization[1:-1] -= cost_gen
        regularization[:-2] += cost_gen
        
    # Calculation by Louis (outdated)
    else:
        # Get normalized bond vector directions
        directions = trace_vectors / trace_norms[:, np.newaxis]
        
        # Get bond offset when compared to chosen segment lenght. The sign is important -> don't use np.abs
        # Positive offset between points p_n and p_{n-1} -> points should move closer together
        # Negative offset -> points should move apart from each other
        bond_offsets = trace_norms - seg_length
    
        # The higher the absolute offset value is, the larger the induced cost is
        # Scale the penalty with an additional penalty term. This penalty term should at ->
        # -> best be *very* large in order to make any gradients pointing in the bond vector ->
        # -> negligible. This corresponds to physical reality.
        cost = (directions.T * bond_offsets).T  * grad_params['beta']
    
        # Get regularization 
        # Explainer: after applying this term, the points could have moved too much,
        # i.e. a former low offset could turn into a high offset.
        # now the point p_n{n-1} gets moved closer to p_n
        # this ensures that the points get only moved w.r.t to the relative offset between
        # its neighbors
        regularization[1:] -= cost
        regularization[:-1] += cost
    
    
    return regularization


def get_angle(vec_1, vec_2):
    """compute the angle between two vectors v1, v2.
    
    Parameters:
    ----------
    vec_1, vec_2: nd-array, shape: (num_points, 2)
       the array-like vectors used to compute the angle
       
    Returns:
    -------
    angle: float
        the angle in degree
    """
    v1_unit = vec_1 / np.linalg.norm(vec_1)
    v2_unit = vec_2 / np.linalg.norm(vec_2)
    
    # Get sign
    #minor = np.linalg.det(np.stack((v1_unit[-2:], v2_unit[-2:])))
    #if minor == 0:
    #    raise NotImplementedError('Parallel vecs')

    # Calculate the dotproduct between the vectors and the angle (rad) afterwards
    dotproduct = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
    angle = np.arccos(dotproduct)
    
    return np.degrees(angle)

def get_angle_cross(vec_1, vec_2):
    """ calculates angle by means of a cross product. If negative, the exterior angle
    is calculated.
    
    Parameters:
    ----------
    vec_1, vec_2: nd-array, shape: (num_points, 2)
       the array-like vectors used to compute the angle
       
    Returns:
    -------
    angle: float
        the angle in radians
    """
    cross_product = np.cross(vec_1, vec_2)
    angle = np.arcsin(cross_product / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2)))
    angle = angle * 180 / np.pi

    return angle

def limit_overfit(trace, grad_params=None):
    """ Method to avoid overfitting due to image effects.
    
    Parameters:
    ----------
    trace: nd-array
        the 'trace' object used to define the direction of displacement
    grad_params: dict
        the dictionary holding the cost penalty assigned to the displacement
        
    Returns:
    --------
    regularization: nd-array
        the increments added to the initial gradients to avoid overfitting
        
    Notes for Leo
    very interesting problem!!
    """
    regularization = np.zeros_like(trace)
    
    # Get angle between bond vectors in rad.
    trace_vecs = np.diff(trace, axis=0)
    angles = np.asarray([get_angle(v1, v2) for v1, v2 in zip(trace_vecs[:-1], trace_vecs[1:])])
    angles = angles[:, np.newaxis] / 180 * np.pi
    
    # Define angle penalty
    angle_penalty = grad_params['lambda'] * np.abs(angles)
     
    # Calculate direction along which the trace points can move
    direction = 2 * trace[1:-1] - (trace[:-2] + trace[2:])
    
    # If the angle is 180 deg.: -> direction = 0, hence + 1e-9
    normed_dir = direction / (np.linalg.norm(direction, axis=1)[:, np.newaxis] + 1e-9)
    
    # Apply the angle penalty -> the trace point gets moved in normed_dir.
    grd_offset = normed_dir * angle_penalty
    
    # Set up regularization gradients pointing in the opposite direction of the offset vector
    regularization[1:-1] -= 0.66 * grd_offset
    
    #########
    # If trace points follow a zig-zag line, moving each angle too much ->
    # -> would invert such a line, thereby creating an inverse problem.
    # To avoid this, check the offset of the neighboring points to see whether ->
    # -> they point in the respective opposite direction and adjust the gradients
    #########
    
    # Move neighbors
    regularization[:-2] += 0.33 * grd_offset # move left neighbors
    regularization[2:] += 0.33 * grd_offset # move right neighbors
    
    return regularization

def limit_grd_displacement(gradients, threshold):
    """ a method that decreases gradients that are above a certain threshold.
    
    Parameters:
    ----------
    gradients: nd_array, shape: (num_points, 2)
        the array of increments calculated for *each* iteration step
        
    threshold: tuple of floats
        the float double indicating the thresholds desired 
        threshold[0] is the upper bound
        threshold[1] is the lower bound
    
    Returns:
    -------
    gradients: the scaled gradients
    
    Example: Let's say all gradients with norm >= 1.5 need to be decreased.
            If we assume that gradient[i] has norm = 1.5, this gradient should
            stay the same as it is exactly at threshold. To counteract the division
            by its norm (1.5), we multiply by the threshold (1.5)
    """
    grd_norms = np.linalg.norm(gradients, axis=1)
    max_mask = grd_norms > threshold[0]
    min_mask = grd_norms < threshold[1]
    
    if np.any(max_mask):
        # scale each gradient by how much its norm exceeds grd_max
        scale_factors =  threshold[0] / (grd_norms[max_mask][:, np.newaxis] + 1e-9)
        gradients[max_mask] *= scale_factors 
        
    if np.any(min_mask):
        # scale each gradient by how its norm subverts grd_min
        scale_factors = threshold[1] / (grd_norms[min_mask][:, np.newaxis] + 1e-9)
        gradients[min_mask] *= scale_factors
    
    return gradients
 

#%% Outdated

def adjust_lengths(trace, grad_params):
    """ by updating the trace, the distances between subsequent trace points are very
    likely to change due to the gradient descent algorithm. If those points stay fixed,
    the lenghts in between them vary, leading to a bad fit. One possible fix is 
    implemented here: if a segment gets too long, it gets splitted in half. if a 
    segment gets too short, it is simply deleted. Doing so enables to keep 
    the lengths at approx. equal length and further avoids overfitting.
    
    An Alternative: 
        We could also include a bigger length penalty, shoulnt this solve the problem?
    But we have two contra-points here: 
        
        SHOULD NOT BE USED
    """
    # get lengths
    vector_norms = np.linalg.norm(np.diff(trace, axis=0), axis=1)
    segment_length = grad_params['segment_length']
    
    # split segments that are too long in half
    for segment_id in np.flip(np.where(vector_norms > segment_length * 1.67)):
        point_new = (trace[segment_id + 1] + trace[segment_id]) / 2 # point to insert
        trace = np.insert(trace, segment_id + 1, point_new, axis=0) # insert at ID
        # print('too big')
    
    # update norms
    vector_norms = np.linalg.norm(np.diff(trace, axis=0), axis=1)
    vector_norms = vector_norms[1:] + vector_norms[:-1] # Why????
    
    # delete segments that are too short
    for segment_id in np.flip(np.where(vector_norms < segment_length * 1.33)): #why 1.33?????        
        trace = np.delete(trace, segment_id + 1, axis=0)
        # print(' too small')
    
    return trace
