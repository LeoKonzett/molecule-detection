# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:43:52 2020

@author: Leo
"""
import analysis_funcs as analysis
import trace_gradient_adaption as trace_ref
import copy
import numpy as np
import interp_funcs

class LinearDNA:

    def __init__(self, afm_molecule, improvement_pars, height_dict=None):

        # Copy the variables, otherwise they are also changed in the AFMMolecule instances
        self.mol_original = copy.deepcopy(afm_molecule.mol_original)
        self.mol_filtered = copy.deepcopy(afm_molecule.mol_filtered)
        self.anal_pars = copy.deepcopy(afm_molecule.anal_pars)
        self.img_meta_data = copy.deepcopy(afm_molecule.img_meta_data)
        self.mol_pars = copy.deepcopy(afm_molecule.mol_pars)

        # Sort the trace obtained by skeletonization
        self.order_trace(improvement_pars)
        
        # If height distribution is known -> update trace
        if height_dict is not None:
            self.update_trace(height_dict, improvement_pars)
            self.get_angles(improvement_pars)
            self.get_length()
            
    def order_trace(self, improvement_pars):
        mol_pars = copy.deepcopy(self.mol_pars)
        
        # Get the skeleton pixel indices
        trace_unsorted = np.transpose(np.nonzero(mol_pars['mol_skel']))
        mol_pars['trace_sorted'] = analysis.order_trace(trace_unsorted)
        
        # Interpolate the trace and select equidistant points along it.
        mol_pars['trace_equidistant'] = interp_funcs.resample_trace(mol_pars['trace_sorted'], 
                            segment_length=improvement_pars['segment_length'], keep_endpoints=False)
        
        # Update the remaining parameters
        self.mol_pars.update({'trace_sorted': mol_pars['trace_sorted'],
                              'trace_equidistant': mol_pars['trace_equidistant']})
        
        return self
            
            
    def update_trace(self, height_dict, improvement_pars):
        mol_pars = copy.deepcopy(self.mol_pars)
        
        # Apply improvement algorithm
        trace_to_refine = mol_pars['trace_equidistant']
        mol_pars['trace_refined'] = trace_ref.refine_trace(
            self.mol_original, trace_to_refine, height_dict, improvement_pars, mol='linear')
        
        # Compare to equidistant trace.
        # Now we want to keep (!) the enpoints since the improvement algorithm optimized it.
        mol_pars['trace_ref_ed'] = interp_funcs.resample_trace(mol_pars['trace_refined'], 
                            segment_length=improvement_pars['segment_length'], keep_endpoints=True)
        
        # Update the dictionary
        self.mol_pars.update({'trace_refined': mol_pars['trace_refined'],
                              'trace_ref_ed': mol_pars['trace_ref_ed']})
        
        return self

    
    def get_angles(self, improvement_pars):
        trace = copy.deepcopy(self.mol_pars['trace_refined'])
        
        # Get segment vectors
        vecs = np.diff(trace, axis=0)
        
        # Get normalized length
        length = np.linalg.norm(vecs, axis=1)
        length /= np.sum(length)
        distance = np.cumsum(length)[:-1]
        
        # Calculate angles
        angles = []
        for v1, v2 in zip(vecs[1:], vecs[:-1]):
            angle_signed = trace_ref.get_angle_cross(v1, v2)
            angles.append(angle_signed)
        
        # Correct counter-clockwise direction
        if sum(angles) < 0:
            angles = [angle * (-1) for angle in angles]
        
        self.mol_pars.update({'angle_value': angles,
                              'angle_location': distance})
        
        return self
    
    def get_length(self):
        mol_pars = copy.deepcopy(self.mol_pars)
        trace = copy.deepcopy(self.mol_pars['trace_refined'])
        
        # Include pixel size
        pixel_size = self.img_meta_data['pixel_size']
        
        # Get length
        dist_vecs = np.linalg.norm(np.diff(trace, axis=0), axis=1)
        mol_pars['length'] = dist_vecs.sum() * pixel_size
        mol_pars['length_etoe'] = np.linalg.norm(trace[0] - trace[-1]) * pixel_size
        
        self.mol_pars.update({'length_total': mol_pars['length'],
                              'length_etoe': mol_pars['length_etoe']})
        
        return self
