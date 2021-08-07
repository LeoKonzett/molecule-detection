# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 10:42:53 2020

@author: Leo
"""
import copy
import numpy as np
import analysis_funcs as analysis
from skimage import morphology, measure
from scipy.ndimage import binary_fill_holes

class AFMMolecule:

    def __init__(self, mol, img_meta_data, anal_pars):
        """ Class to identify different molecule types. First, a 1 pixel wide skeleton of a given molecule 
        is calculated. Then, the molecule is classified accordingto its properties.
        Those properties are outlined in the docs of the mol_props function.
        
        Attributes:
        ----------
        mol_original: nd-array
            originial, unfiltered image of a DNA strand.
        mol_filtered: nd-array
            filtered image of a DNA strand.
        img_meta_data: dict
            holds various parameters such as pixel_size.
        anal_pars: dict
            holds parameters for image analysis, outlined in run_code file.
        mol_pars: dict
            enables to add various parameters.
        
        Adapted from Sebastian Konrad.
        """
        self.mol_original = mol['mol_original']
        self.mol_filtered = mol['mol_filtered']
        self.img_meta_data = img_meta_data
        self.anal_pars = anal_pars
        self.mol_pars = {'mol_bbox': mol['mol_bbox']}

        # Skeletonize the molecule
        self.mol_props(anal_pars['skeletonize_method'], do_pruning=anal_pars['do_pruning'])
        
        # Categorize the molecule
        self.categorize_molecules_updated()
        
        
    def mol_props(self, method='lee', do_pruning=True):
        """ Calculates the following properties of a skeletonized molecule:
            
            1) mol_width: the maximum distance out of all molecule trace points before and after 
                        applying a medial axis transform. (Outdated)
            2) mol_skel: the 1-pixel wide skeleton of the molecule.
            3) max_area_over_height: maximum area of connected pixels above a given height.
            4) skel_eps_number, skel_bps_number: number of skeleton endpoints / branchpoints.
            5) sel_pixel_number: number of pixels comprising the skeleton.
        
        Parameters:
        ----------
        method: str
            indicates the skeletonization algorithm employed.
        
        do_pruning: bool
            indicates whether or not to delete skeleton branches that result from incomplete 
            skeletonization. See analyis_funcs module for details.
            
        Returns:
        -------
        mol_pars: dict
            updated mol_pars dictionary holding all the parameters discussed above (mol_width, etc.)
            
        Note:
        -----
        It is advised to ONLY use the pruning algorithm with a very short maximum pruning length.
        Otherwise, subsequent tracing errors may occur. The process was optimized in such a way that pruning is 
        almost never necessary, since faulty branchpoints occur only about 1 out of 50 times.
        """
        mol_bw = copy.deepcopy(self.mol_filtered)
        mol_filtered = copy.deepcopy(self.mol_filtered)
        
        # Convert to binary and fill connected objects.
        mol_bw[mol_bw > 0] = 1
        mol_filled = binary_fill_holes(mol_bw)
        
        # Get maximum thinning distance by medial_axis transform
        _, thin_distance = morphology.medial_axis(mol_filled, return_distance=True)
        
        # Maximum thinning distance corresponds to teardrop radius
        mol_width = np.max(thin_distance)
        
        # Skeletonize with method Lee (yielded the best results)
        mol_skel = morphology.skeletonize(mol_bw, method=method).astype(bool)
        
        # If do_pruning is True, faulty branchpoints from skeletonization are deleted
        if do_pruning:
            _, skel_eps_pixels = analysis.get_junctions(mol_skel)
            mol_skel = analysis.prune_skeleton(mol_skel, skel_eps_pixels)
        
        # Max_area_over_height - calculate the largest area of connected pixels with a value over self.nuc_min_height
        if np.amax(mol_filtered) > self.anal_pars['strep_min_height']:
            mol_over_height = copy.deepcopy(mol_filtered)
            mol_over_height[mol_over_height < self.anal_pars['strep_min_height']] = 0
            mol_over_height[mol_over_height > self.anal_pars['strep_min_height']] = 1
            img_labelled = morphology.label(mol_over_height, connectivity=2)
            max_area_over_height = max(region.area for region in measure.regionprops(img_labelled) if region.area)
        else:
            max_area_over_height = 0
        
        # Update dictionary
        self.mol_pars['mol_width'] = mol_width
        self.mol_pars['mol_skel'] = mol_skel
        self.mol_pars['max_area_over_height'] = max_area_over_height
        
        # Update dictionary with skeleton branch / endpoints
        self.mol_pars.update(analysis.get_skel_pars(self.mol_pars['mol_skel']))
        
    def categorize_molecules_updated(self):
        """ method to categorize a molecule according to its properties. All molecules that 
        cannot be classified are assigned to various trash bins. The categorization algorithm 
        checks for: 
            1) number of pixels comprising the skeleton
            2) number of branch / endpoints
            3) area of connected pixels above a given height (indicating strep. complex)

        Returns:
        -------
        mol_pars: dict (update)
            adds a Type variable indicating the type of the molecule. Possible types are
            Linear, Teardrop, Dimer, Potential Linear, Potential Teardrop, Trash.

        """
        # Calculate number of expected pixels to filter very large / small molecules
        mol_pars = self.mol_pars
        anal_pars = self.anal_pars
        expected_pixels = anal_pars['dna_length_bp'] * 0.33 / self.img_meta_data['pixel_size']

        # Undersized skeleton
        if mol_pars['skel_pixels_number'] <= expected_pixels / 3:
            mol_pars['type'] = 'Trash'
            mol_pars['reason'] = 'Skeleton too small'
            
        # Oversized skeleton - potentially connected good molecules
        elif mol_pars['skel_pixels_number'] > expected_pixels * 1.5: 
            mol_pars['type'] = 'Potential'
            mol_pars['reason'] = 'Large skeleton'

        # Linear DNA
        elif mol_pars['skel_eps_number'] == 2 and mol_pars['skel_bps_number'] == 0 \
            and mol_pars['max_area_over_height'] == 0:
            mol_pars['type'] = 'Bare DNA'

        # Linear DNA with branch
        elif mol_pars['skel_eps_number'] <= 4 and mol_pars['skel_bps_number'] in [1, 2]:
            mol_pars['type'] = 'Potential Bare DNA'
            mol_pars['reason'] = 'up to 4 EPS, up to 2 BP'
        
        # Dimer DNA
        elif mol_pars['skel_eps_number'] == 2 and mol_pars['skel_bps_number'] == 0 \
            and mol_pars['max_area_over_height'] != 0:
            mol_pars['type'] = 'Potential Dimer'
            mol_pars['reason'] = 'Contains streptavidin'
        
        # Teardrop DNA
        elif mol_pars['skel_eps_number'] == 0 and mol_pars['skel_bps_number'] == 0:
            mol_pars['type'] = 'Teardrop DNA'
        
        # Teardrop DNA with branch
        elif mol_pars['skel_eps_number'] <= 2 and mol_pars['skel_bps_number'] in [1, 2]:
            mol_pars['type'] = 'Potential Teardrop DNA'
            mol_pars['reason'] = 'up two 2 EPS, up to two BP'
        
        # Undefined trash
        else:
            mol_pars['type'] = 'Trash'
            mol_pars['reason'] = 'Undefined'
        
        # Update dictionary
        self.mol_pars.update(mol_pars)

        return

#%% Outdated    
    def categorize_molecules(self):
        """ method to categorize molecules according the outlined parameters.

        Returns:

        """
        # Skeleton parameters
        self.mol_pars.update(analysis.get_skel_pars(self.mol_pars['mol_skel']))
        mol_pars = self.mol_pars
        anal_pars = self.anal_pars
        expected_pixels = anal_pars['dna_length_bp'] * 0.33 / self.img_meta_data['pixel_size']

        # Undersized skeleton
        if mol_pars['skel_pixels_number'] <= expected_pixels / 3: # prev. 2
            mol_pars['type'] = 'Trash'
            mol_pars['reason'] = 'Skeleton too small'
            
        # Oversized skeleton - potentially connected good molecules
        elif mol_pars['skel_pixels_number'] > expected_pixels * 1.8: #prev. 1.5
            mol_pars['type'] = 'Potential'
            mol_pars['reason'] = 'Large skeleton'

        # Linear DNA
        elif mol_pars['skel_eps_number'] == 2 and mol_pars['skel_bps_number'] == 0: #\
            #and mol_pars['mol_width'] < anal_pars['mol_width']:
            mol_pars['type'] = 'Bare DNA'

        # Bare DNA with branch
        elif mol_pars['skel_eps_number'] <= 4 and mol_pars['skel_bps_number'] in [1, 2]: #\
            #and mol_pars['mol_width'] < anal_pars['mol_width']:
            mol_pars['type'] = 'Potential bare DNA'
            mol_pars['reason'] = 'up to 4 EPS, up to 2 BP'
        
        # Teardrop DNA
        elif mol_pars['skel_eps_number'] == 0 and mol_pars['skel_bps_number'] == 0: #\
            #and mol_pars['mol_width'] > anal_pars['mol_width']:
            mol_pars['type'] = 'Teardrop DNA'
        
        # Teardrop DNA with branch
        elif mol_pars['skel_eps_number'] <= 2 and mol_pars['skel_bps_number'] in [1, 2]: #\
            #and mol_pars['mol_width'] > anal_pars['mol_width']:
            mol_pars['type'] = 'Potential Teardrop DNA'
            mol_pars['reason'] = 'up two 2 EPS, up to two BP'
        
        # Undefined trash
        else:
            mol_pars['type'] = 'Trash'
            mol_pars['reason'] = 'Undefined'
        
        """
        elif mol_pars['skel_bps_number'] >= 1 and mol_pars['mol_width'] > anal_pars['mol_width']:
            mol_pars['type'] = 'Trash'
            mol_pars['reason'] = 'Teardrop, too many BPS'
        
        elif mol_pars['skel_bps_number'] >=2 and mol_pars['mol_width'] < anal_pars['mol_width']:
            mol_pars['type'] = 'Trash'
            mol_pars['reason'] = 'Linear, too many BPS'
        """

        self.mol_pars.update(mol_pars)

        return
