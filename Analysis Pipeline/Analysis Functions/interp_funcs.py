# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 13:19:10 2020

@author: Leo
"""
import numpy as np
from scipy.linalg import norm
from scipy.interpolate import interp1d
import math

def interpolate_trace(trace, int_factor=400, method='cubic'):
    """ Function to interpolate the coordinates of a 2d DNA backbone trace.

    Parameters:
    ----------
    trace: array of int64
        input array of shape (num_coord_points, 2)
    int_factor: int
        the number of interpolation points per segment
    method: str
        the method used for interpolation (linear, quadratic, cubic)

    Returns:
    -------
    interp_coords: array of float
        array of shape (N * int_factor, 2) holding the interpolated coords
    """
    num = len(trace) - 1 # -1 because of index: (0, 1, 2) has length 3 but only index 2
    x_data = np.arange(num + 1) # get points on x axis up to num
    step_num = num * int_factor # get number of points to interpolate
    x_new = np.linspace(0, num, step_num + 1)

    # Interpolate for x(0) and y(1) coordinates
    f_x = interp1d(x_data, trace.T[0], kind=method)
    f_y = interp1d(x_data, trace.T[1], kind=method)
    interp_coords = np.vstack((f_x(x_new), f_y(x_new))).T
    
    return interp_coords

def resample_trace(trace, segment_length=2.5, keep_endpoints=True, 
                          method='cubic', iterative=False, int_factor=15):
    """ Resamples a trace representing the DNA backbone obtained by skeletonization. Selects
    those coordinate points that are *at least* segment_length apart. The distance between
    two trace points is measured using the coordinate-wise euclidean norm. If keep_endpoints is True,
    the *last* point of the backbone does not change.
    
    Parameters:
    -----------
    trace: array of int64
        input array of shape (num_coord_points, 2)
    segment_length: float
        the segment length used for splitting
    keep_endpoints: bool
        indicates whether or not to keep trace endpoints
    method: str
        the interpolation method used  (np.interp1d)

    Returns:
    -------
    trace_equidistant: array of float
        the equidistant trace
    
    """
    # Interpolate the trace -> cubic
    trace_fine = interpolate_trace(trace, int_factor=int_factor, method=method) # prev. 400
    
    # Iterative not advised to use (!) -> implement in numba if time
    # Get equidistant points by iterative calculation -> more accurate, but computationally expensive
    if iterative:
        # Initialize list
        i = 0
        if keep_endpoints:
            index_list = [0]
        else:
            index_list = []
        while i < len(trace_fine):
            total_dist = 0
            for j in range(i+1, len(trace_fine)):
                total_dist += math.sqrt((trace_fine[j][0] - trace_fine[j-1][0])**2 
                                        + (trace_fine[j][1] - trace_fine[j-1][1])**2)
                if total_dist > segment_length:
                    index_list.append(j)
                    break
                i = j + 1
                
        if keep_endpoints:
            index_list.append(-1)
    
    # Faster calculation implemented in numpy
    else:
        # Get arc length of interpolated trace
        distance = np.cumsum(norm(np.diff(trace_fine, axis=0), axis=1))
    
        # Account for difference between arc length and segment length
        arc_length = segment_length * (1 + 0.05)
        distance = np.mod(distance, arc_length)
        distance = np.append([0], distance) # Correction due to np.diff
        
        ##########
        # Example: [0, 1, 2, 0, 1, 2] (after mod. division) ->
        # -> indicates that element[i=3] is larger than segment_length
        # Two options: either select the element[i=3] > segment_length ->
        # -> or element[i=2] < segment_length
        ##########
        
        # Set boolean mask to select equidistant points from the interpolated trace
        length_diffs = np.diff(distance, axis=0)
        passed_sl = length_diffs < 0
        
        # Append starting point
        index_list = np.append([True], passed_sl)
        
        # If True, the last trace point is included in the boolean mask
        if keep_endpoints:
            index_list[-1] = True 
        
    return  trace_fine[index_list]


def resample_strep(this_trace, thresh=8, num=15):
    """ Method to determine the points where the DNA arms enter the binding site. 
    Depending on the chosen strep. radius, the first / second / third trace point 
    is connected to the streptavidin center. The resulting segment is interpolated. 
    From the interpolated points, exactly that point is chosen whose distance
    to the center matches the strep. radius. 
    
    Parameters:
    ----------
    this_trace: nd-array
        the optimized trace. trace[0], trace[-1] marks the location of the strep. center
    thresh: float
        the streptavidin radius
    num: int
        the number of points to interpolate
        
    Returns:
    -------
    this_trace: nd-array
        the optimized trace object. The points where the DNA arms enter the binding site
        are always (!) at index [1, -2]
    """
    # Get norm of each segment
    segment_norms = np.linalg.norm(np.diff(this_trace, axis=0), axis=1)
    
    # Problem: First / last segment norm and thresh can be approx. equal ->
    # -> set lower bound
    lower_bound_seg_f = segment_norms[0] - 1.5
    lower_bound_seg_l = segment_norms[-1] - 1.5
    
    # Check if lower bound is larger than thresh
    if thresh < lower_bound_seg_f:
        # Interpolate and resample the first segment
        seg_1_resampled = resample_trace(this_trace[:2], int_factor=num, method='linear',
                                segment_length=thresh, keep_endpoints=False)
        
        # Get point where DNA arm enters streptavidin
        point_intersection_f = seg_1_resampled[1]
        
        # Insert that point into the refined trace
        this_trace = np.insert(this_trace, 1, point_intersection_f, axis=0)
            
    # Check if combind norm of first two segments is larger than thresh
    elif thresh < (lower_bound_seg_f + segment_norms[1]): 
        # Interpolate and resample the first two segments
        seg_1_resampled = resample_trace(this_trace[:3], int_factor=num, method='linear',
                        segment_length=thresh, keep_endpoints=False)
        
        # Get intersection point
        point_intersection_f = seg_1_resampled[1]
        
        # Insert intersection point
        this_trace = np.delete(this_trace, 1, axis=0)
        this_trace = np.insert(this_trace, 1, point_intersection_f, axis=0)
        
    else: # Safety case for segment length 10 nm.
        seg_1_resampled = resample_trace(this_trace[:4], int_factor=num, method='linear',
                        segment_length=thresh, keep_endpoints=False)
        point_intersection_f = seg_1_resampled[1]
        this_trace = np.delete(this_trace, [1, 2], axis=0)
        this_trace = np.insert(this_trace, 1, point_intersection_f, axis=0)
        
    # Flip array
    this_trace = np.flip(this_trace, axis=0)
        
    # Repeat all steps from above
    # Check if lower bound is larger than thresh
    if thresh < lower_bound_seg_l:
        # Interpolate and resample the last segment
        seg_1_resampled = resample_trace(this_trace[:2], int_factor=num, method='linear',
                        segment_length=thresh, keep_endpoints=False)
        
        point_intersection_f = seg_1_resampled[1]
        this_trace = np.insert(this_trace, 1, point_intersection_f, axis=0)
    
    # Check if thresh is smaller than combined norm of last two segments
    elif thresh < (lower_bound_seg_l + segment_norms[-2]): 
        # Interpolate and resample the last two segments
        seg_1_resampled = resample_trace(this_trace[:3], int_factor=num, method='linear',
                        segment_length=thresh, keep_endpoints=False)
        
        # Get intersection point
        point_intersection_l = seg_1_resampled[1]
        this_trace = np.delete(this_trace, 1, axis=0)
        this_trace = np.insert(this_trace, 1, point_intersection_l, axis=0)
        
    else: # Safety case
        seg_1_resampled = resample_trace(this_trace[:4], int_factor=num, method='linear',
                        segment_length=thresh, keep_endpoints=False)
        point_intersection_l = seg_1_resampled[1]
        this_trace = np.delete(this_trace, [1, 2], axis=0)
        this_trace = np.insert(this_trace, 1, point_intersection_l, axis=0)
    
    return this_trace


#%% Outdated

def set_equidistant_trace_2(trace, segment_length=2.5, int_factor=10, keep_endpoints=True):
    """
    Intermediary note: for linear, it serves to remove the endpoints.
    For teardrops, it smoothens the drops quite a bit and removes zaggy edges.
    One point to highlight: The streptavidin doesn't necessarily have to be close to the skeleton
    So one should think of a method to remove the trace points next to the streptavidin.

    Parameters:
    -----------
    trace: array of int64
        input array of shape (num_coord_points, 2) representing the skeleton trace
    segment_length: float
        the segment length in used for splitting
    mol: str
        the molecule to analyze: linear or teardrop

    Returns:
    -------
    int_trace: array of float
        coordinate array with those points selected that are approx. segment_length apart.
    """
    # Smaller segment length -> more interpolation points
    # int_factor = np.ceil(1000 / segment_length).astype(int)
#    trace_fine = interpolate_trace(trace, int_factor=int_factor, method=method)
    
    # Initialize list
    i = 0
    index_list = [0]
    while i < len(trace_fine):
        total_dist = 0
        for j in range(i+1, len(trace_fine)):
            total_dist += math.sqrt((trace_fine[j][0] - trace_fine[j-1][0])**2 
                                    + (trace_fine[j][1] - trace_fine[j-1][1])**2)
            if total_dist > segment_length:
                index_list.append(j)
                break
        i = j + 1
    
    index_list.append(-1)
    trace_equidistant = trace_fine[index_list]
    
    return trace_equidistant

