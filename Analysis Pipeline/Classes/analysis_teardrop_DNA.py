# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:14:28 2020

@author: Leo
"""
import analysis_funcs as analysis
from skimage import filters, morphology, measure
import trace_gradient_adaption as trace_ref
import copy
import numpy as np
import interp_funcs

class TeardropDNA:
    """class docstring
    """

    def __init__(self, afm_molecule, improvement_pars, height_dict=None):
        # Copy the variables, otherwise they are also changed in the AFMMolecule instances
        self.mol_original = copy.deepcopy(afm_molecule.mol_original)
        self.mol_filtered = copy.deepcopy(afm_molecule.mol_filtered)
        self.anal_pars = copy.deepcopy(afm_molecule.anal_pars)
        self.img_meta_data = copy.deepcopy(afm_molecule.img_meta_data)
        self.mol_pars = copy.deepcopy(afm_molecule.mol_pars)

        # Order the backbone trace obtained by skeletonization and identify streptavidin
        self.locate_streptavidin()
        self.order_trace(improvement_pars)

        if height_dict is not None:
            # Update the trace + streptavidin blob
            self.update_strep(height_dict, improvement_pars)
            
            # Get properties
            self.get_angles(improvement_pars)
            self.get_length()
        
    def order_trace(self, improvement_pars):
        mol_pars = copy.deepcopy(self.mol_pars)
        
        # Sort the DNA backbone obtained by skeletonization
        trace_unsorted = np.transpose(np.nonzero(mol_pars['mol_skel']))
        mol_pars['trace_sorted'] = analysis.order_teardrop_trace(
            trace_unsorted, mol_pars['strep_index'])
        
        # Interpolate the trace and select equidistant points along it.
        mol_pars['trace_equidistant'] = interp_funcs.resample_trace(
            mol_pars['trace_sorted'], segment_length=improvement_pars['segment_length'], keep_endpoints=True)
        
        self.mol_pars.update({'trace_sorted': mol_pars['trace_sorted'], 
                              'trace_equidistant': mol_pars['trace_equidistant']})
        
        return self
    
    def locate_streptavidin(self):
        """ Method to locate the center of the binding site complex.
        """
        mol_pars = copy.deepcopy(self.mol_pars)
        mol_filtered = copy.deepcopy(self.mol_filtered)
        
        # Apply Gaussian filter to facilitate a bimodal histogram split
        img_gaussian = filters.gaussian(mol_filtered, sigma=1.)
        
        # Check maximum intensity pixel index
        max_index = np.unravel_index(np.argmax(img_gaussian), img_gaussian.shape)
        
        # Get threshold (yen threshold) for strep. complex
        if self.anal_pars['strep_min_height'] is None:
            thresh = filters.threshold_yen(img_gaussian)
            
            # If Yen thresholding fails try Minimum thresholding:
            if thresh < 0.8:
                thresh = filters.threshold_minimum(img_gaussian)
                
        else: # Alternative option is setting threshold manually
            thresh = self.anal_pars['strep_min_height']
            
        # Split the image according to threshold selected
        img_bw = copy.deepcopy(img_gaussian)
        img_bw[img_bw < thresh] = 0
        img_bw[img_bw != 0] = 1
         
        # Label the regions that lie above the threshold (i.e. the streptavidin)
        img_labelled = morphology.label(img_bw)
        
        # Check if the thresholding yielded more than one area:
        if img_labelled.max() != 1:
            
            # Remove possible artefacts (sometimes happens for Yen filter)
            img_bw = morphology.remove_small_objects(img_bw.astype(bool))
            img_labelled = morphology.label(img_bw) # relabel image
        
        # Loop through regions
        for region in measure.regionprops(img_labelled):
            
            # Get index of strep. complex by centroid
            # Check if centroid is near max_index to ensure correct tracing
            if np.linalg.norm(np.asarray(max_index) - region.centroid) < 0.2:
                mol_pars['strep_index'] = region.centroid
            else:
                mol_pars['strep_index'] = max_index
            
            # Calculate the major and minor axis of the strep. complex to calculate approx. radius
            y0, x0 = mol_pars['strep_index'][0], mol_pars['strep_index'][1]
            orientation = region.orientation
            x1 = x0 + np.cos(orientation) * 0.5 * region.minor_axis_length
            y1 = y0 - np.sin(orientation) * 0.5 * region.minor_axis_length
            x2 = x0 - np.sin(orientation) * 0.5 * region.major_axis_length
            y2 = y0 - np.cos(orientation) * 0.5 * region.major_axis_length
            mol_pars['strep_minor_axis'] = (x1, y1)
            mol_pars['strep_major_axis'] = (x2, y2)
            mol_pars['axis_length'] = (region.major_axis_length + region.minor_axis_length) * 0.5
            
        # Add the streptavidin area to the mol_pars dictionary
        img_strep = copy.deepcopy(img_bw)
        mol_pars['img_strep'] = img_strep
        
        self.mol_pars.update({'strep_index': mol_pars['strep_index'],
                              'strep_minor_axis': mol_pars['strep_minor_axis'],
                              'strep_major_axis': mol_pars['strep_major_axis'],
                              'axis_length': mol_pars['axis_length'],
                              'img_strep': mol_pars['img_strep']})
        
        return self
            
    
    def update_strep(self, height_dict, improvement_pars):
        """method to apply optimization algorihtm to teardrop-like strands
        """
        mol_original = copy.deepcopy(self.mol_original)
        mol_pars = copy.deepcopy(self.mol_pars)
        
        # Set the trace to be improved
        trace_to_improve = mol_pars['trace_sorted']
        origin = mol_pars['strep_index']
        
        # Set clip radius
        clip_radius = improvement_pars['strep_diameter'] + 1
        
        ########
        # Refine molecule wo/ strep complex
        ########
        
        # Delete pixel values within circle that has its origin at strep. location
        img_strep_deleted = analysis.in_circle(mol_original, clip_radius, 
                                        origin, target_val=0)
        
        # Interpolate the input trace to ensure minimal cutting error
        int_factor = improvement_pars['steps_per_pixel']
        trace_fine = interp_funcs.interpolate_trace(trace_to_improve, int_factor)
        
        # Get coordinate-wise euclidean distance of each interpolated trace point -> 
        # -> to streptavidin complex
        dist_to_strep = np.linalg.norm(trace_fine - origin, axis=1)
        
        # Delete all trace points that fall into the circle defined above
        trace_strep_deleted = trace_fine[dist_to_strep > clip_radius]
        
        # Spread points evenly along the cut-off trace before applying improvement algorithm
        mol_pars['trace_sd'] = interp_funcs.resample_trace(trace_strep_deleted,
                                        improvement_pars['segment_length'], keep_endpoints=False)
        
        # Apply improvement algorithm to cut-off trace, i.e. treat molecule as linear
        trace_sd_ref = trace_ref.refine_trace(
            img_strep_deleted, mol_pars['trace_sd'], height_dict, improvement_pars, mol='linear')
        
        # Re-add streptavidin; sd: strep deleted
        trace_sd_ref = np.append([mol_pars['strep_index']], trace_sd_ref, axis=0)
        trace_sd_ref = np.append(trace_sd_ref, [mol_pars['strep_index']], axis=0)
        mol_pars['trace_sd_refined'] = trace_sd_ref
        
        # Resample strep; si: strep included
        trace_si_refined = copy.deepcopy(trace_sd_ref)
        mol_pars['trace_si_refined'] = interp_funcs.resample_strep(trace_si_refined,
                                                       thresh=improvement_pars['strep_diameter'])
        
        # Update the dictionary
        self.mol_pars.update({'trace_sd': mol_pars['trace_sd'],
                              'trace_sd_refined': mol_pars['trace_sd_refined'],
                              'trace_si_refined': mol_pars['trace_si_refined']})
        
        return self
    
 
    def get_angles(self, improvement_pars):
        trace = copy.deepcopy(self.mol_pars['trace_si_refined'])
        
        # Get bond vectors
        vecs = np.diff(trace, axis=0)
        
        # Get normalized length
        length = np.linalg.norm(vecs, axis=1)
        length /= np.sum(length)
        distance = np.cumsum(length)[:-1]
        
        # Exit angle between triangle out of: strep, first trace point, last trace point
        vec_1 = trace[1] - trace[0]
        vec_2 = trace[-2] - trace[-1]
        exit_angle = trace_ref.get_angle(vec_1, vec_2)
        if exit_angle < 0:
            exit_angle *= -1
        
        # Angle between 2nd and 2nd-last segment of the optimized DNA backbone
        vec_3 = trace[2] - trace[1]
        vec_4 = trace[-3] - trace[-2]
        
        exit_angle_seg = trace_ref.get_angle(vec_3, vec_4)
        if exit_angle_seg < 0:
            exit_angle_seg *= -1
        
        # Calculate other angles
        angles = []
        for v1, v2 in zip(vecs[1:], vecs[:-1]):
            angle_signed = trace_ref.get_angle_cross(v1, v2)
            angles.append(angle_signed)
        
        # Correct counter-clockwise direction
        if sum(angles) < 0:
            angles = [angle * (-1) for angle in angles]
        
        # Update dictionary
        self.mol_pars.update({'angle_value': angles,
                              'angle_location': distance,
                              'exit_angle_strep': exit_angle,
                              'exit_angle_segment': exit_angle_seg})
       
        return self
    
    def get_length(self):
        mol_pars = copy.deepcopy(self.mol_pars)
        trace = copy.deepcopy(self.mol_pars['trace_sd_refined'])
        pixel_size = self.img_meta_data['pixel_size']
        
        dist_vecs = np.linalg.norm(np.diff(trace, axis=0), axis=1)
        mol_pars['length'] = dist_vecs.sum() * pixel_size
        
        self.mol_pars.update({'length_total': mol_pars['length']})
        
        return self
    
#%% Outdated
"""
def refine_strep(self, tip_dict, improvement_pars):
        
        mol_pars = copy.deepcopy(self.mol_pars)
        
        if 'trace_refined_2' in mol_pars.keys():
            trace_to_improve = mol_pars['trace_refined_2']
            strep_diameter = improvement_pars['strep_diameter']
            
        else:
            trace_to_improve = mol_pars['trace_refined']
            strep_diameter = improvement_pars['strep_diameter'] + 3.
            
        # Interpolate the refined trace to ensure minimal cutting error.
        # Smaller segment length -> more interpolation points
        int_factor = np.ceil(100 / improvement_pars['segment_length']).astype(int)
        trace_fine = interp_funcs.interpolate_trace(trace_to_improve, int_factor)
        
        # Get distance from streptavidin
        dist_to_strep = np.linalg.norm(
            trace_fine - mol_pars['strep_index'], axis=-1)
        
        # Check for circle with radius strep_diameter
        trace_pruned = trace_fine[dist_to_strep > strep_diameter]
        
        # Add Streptavidin again (filtered out by circle)
        trace_pruned = np.append([mol_pars['strep_index']], trace_pruned, axis=0)
        trace_pruned = np.append(trace_pruned, [mol_pars['strep_index']], axis=0)
        
        # Set equidistant points
        mol_pars['trace_pruned'] = interp_funcs.smoothen_trace(
            trace_pruned, segment_length=improvement_pars['segment_length'], mol='teardrop')
        
        # Get Exit angle
        # Define opening vectors
        vec_1 = trace_pruned[1] - mol_pars['strep_index']
        vec_2 = trace_pruned[-2] - mol_pars['strep_index']
        # Get angle in degree
        mol_pars['exit_angle'] = trace_ref.get_angle(vec_1, vec_2)
        
        # Update the dictionary
        self.mol_pars.update({'trace_pruned': mol_pars['trace_pruned'],
                              'exit_angle': mol_pars['exit_angle']})
        
        return self
    
def cut_streptavidin(self, improvement_pars):
        
        mol_original = copy.deepcopy(self.mol_original)
        mol_pars = copy.deepcopy(self.mol_pars)
        
        # Clip pixel values that are bigger than the tip shape center
        # img_capped = copy.deepcopy(self.mol_filtered)
        # tip_center_height = tip_dict['tip_shape_fct'](0,0) * 1.1
        # img_capped = np.clip(img_capped, None, tip_center_height)
        
        # Set the trace to be improved
        trace_to_improve = mol_pars['trace_refined']
        
        # Set pixel values within a circle def. by strep. diameter to zero
        # The origin of the cirlce is defined by the location of the strep. complex
        max_index = mol_pars['strep_index']
        img_strep_deleted = analysis.in_circle(
            mol_original, improvement_pars['strep_diameter'], max_index)
        
        # Cut off all coordinate points in trace_refined that lie within the defined circle
        # Interpolate the trace to ensure minimal cutting error
        int_factor = np.ceil(1000 / improvement_pars['segment_length']).astype(int)
        trace_fine = interp_funcs.interpolate_trace(trace_to_improve, int_factor)
        
        # Get euclidean distance of interpolated trace points to streptavidin complex
        dist_contour = np.cumsum(np.linalg.norm(np.diff(trace_fine, axis=0), axis=1))
        dist_contour = np.append([0], dist_contour)
        dist_contour_inverse = np.max(dist_contour) - dist_contour
        
        # Delete all trace points that lie within a circle def. by strep. diameter
        trace_strep_deleted = trace_fine[dist_contour > improvement_pars['strep_diameter']]
        trace_Step_deleted = trace_fine[dist_contour_inverse > improvement_pars['strep_diameter']]
        
        # Spread points evenly along the cut-off trace before applying improvement algorithm
#        trace_pruned = interp_funcs.smoothen_trace(trace_pruned,
 #                                                  improvement_pars['segment_length'], mol='teardrop')
        return None
    
     def update_strep_2(self, tip_dict, improvement_pars):
        mol_original = copy.deepcopy(self.mol_original)
        mol_pars = copy.deepcopy(self.mol_pars)
        
        # Clip pixel values that are bigger than the tip shape center
        img_capped = copy.deepcopy(self.mol_filtered)
        tip_center_height = tip_dict['tip_shape_fct'](0,0) * 1.1
        img_capped = analysis.in_circle(mol_original, improvement_pars['strep_diameter'], 
                                        max_index, target_val=tip_center_height)
        
        # Set the trace to be improved
        trace_to_improve = mol_pars['trace_equidistant']
        
        # Apply improvement algorithm
        trace_sd_ref = trace_ref.refine_trace(
            img_capped, trace_to_improve, tip_dict, improvement_pars, mol='teardrop')
        
        # Update dictionary
        self.mol_pars.update({'trace_sd_refined': mol_pars['trace_sd_refined']})
        
        #####
        # Part for exit angle
        #####
        
        
            
        # Get euclidean distance of each interpolated trace point to streptavidin complex
        dist_to_strep = np.linalg.norm(trace_fine - max_index, axis=1)
        
        # Delete all trace points that lie within a circle def. by strep. diameter
        trace_strep_deleted = trace_fine[dist_to_strep > improvement_pars['strep_diameter']]
    
    def update_trace(self, tip_dict, improvement_pars):
        
        mol_pars = copy.deepcopy(self.mol_pars)
        
        ###### Large Q
        # Clip pixel values that are bigger than the tip shape center
        #img_capped = copy.deepcopy(self.mol_filtered)
        #tip_center_height = tip_dict['tip_shape_fct'](0,0) * 1.1
        #img_capped = np.clip(img_capped, None, tip_center_height)
        #######
        
        # Apply improvement algorithm
        trace_to_refine = mol_pars['trace_equidistant']
        mol_pars['trace_refined'] = trace_ref.refine_trace(
            self.mol_original, trace_to_refine, tip_dict, improvement_pars, mol='teardrop')
        
        # Update the dictionary
        self.mol_pars.update({'trace_refined': mol_pars['trace_refined']})
    
        
        #if 'trace_pruned' in mol_pars.keys():
        #    # Do another refinement to avoid "zaggy edges"
        #    trace_to_refine = mol_pars['trace_pruned']
        #    mol_pars['trace_refined_2'] = trace_ref.refine_trace(
        #        img_capped, trace_to_refine, tip_dict, improvement_pars, mol='teardrop')
            
            # Update the dictionary
        #    self.mol_pars.update({'trace_refined_2': mol_pars['trace_refined_2']})

        
        return self
    """