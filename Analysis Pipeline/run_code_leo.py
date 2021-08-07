# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:53:41 2020

@author: Leo
"""
#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import import_data
import mol_categorization as cat
import image_processing
import height_estimation_pipeline as hep
import analysis_linear_DNA
import analysis_teardrop_DNA
import interp_funcs
import trace_gradient_adaption as trace_ref
from scipy.stats import norm
import matplotlib.mlab as mlab


#%% RUN CODE

# Get directory
directory = 'C:/Users/leoko/OneDrive/Desktop/recovery/new images/500 bp'
file_list = import_data.get_file_list(directory)


anal_pars = {'manual_thresh': None,         # Threshold value to delete pixel background -> None
             'mol_width': 10,               # Threshold value to separate teardrops from linear molecules. Outdated but maybe useful for further appl.
             'strep_min_height': 1.50,      # Threshold value to detect streptavidin
             'mol_min_area': 1300,          # Minimum amount of pixels for a molecule to be considered for analysis -> 500bp
             #'mol_min_area': 700,          # Same but for 250bp
             'thresh_method': 'otsu',       # Method used to threshold image
             'filter_method': 'median',     # Filter to smooth molecule boundary lines. Not advised to change.
             'dna_length_bp': 486,          # DNA base pair length
             'skeletonize_method': 'lee',   # Method used to skeletonize molecule. Not advised to change.
             'do_pruning': False}           # If True, small skeleton branches can be removed.
                                            # Attention: Should not be necessary because of updated filter mask.

gradient_parameters = {'set_equidistant': True, # Necessary to keep trace points equidistant
                       'limit_overfit': True, # Necessary to avoid overfitting
                       'beta': 2., # Length regularization parameter
                       'lambda': 1., # Overfit regularization parameter
                       'new_gradients_fine': True, # Adaption to gradient calculation
                       'lr_updated': True} # Adaption to length regularization


improvement_parameters = {'max_radius': 8, # Max. distance for pixel intensity distribution function 
                          'ref_radius': 3., # Max. distance up to which pixels are considered for optimization
                          'num_iter': 100, # Number of Iterations performed
                          'strep_diameter': 5, # Radius of streptavidin
                          'steps_per_pixel': 15, # fix 02.04
                          'segment_length': 2.5, # Segment length - > always used to calculate pixel distribution function
                          'points_to_fix': [0, -1], # Indices of trace points that do not change during otpimization
                          'learning_rate': 0.02, # Learning rate
                          'gradient_thresholds': (1, 0), # Set limits for trace displacement
                          'allow_segment_breakup': False,  # Allows breakup -> Outdated, DO NOT USE!
                          'grad_params': gradient_parameters
                          }

#%%
#"C:/Users/Leo/Desktop/Bachelor Thesis/new images/save-2020.11.25-15.01.58.690-3.jpk.asc"
# Import the desired .ascii file (manual selection)

def process_pipeline(anal_pars, improvement_parameters, file_list):
    """ Method to run tracing optimization. """
   
    # Store optimized molecules in a list
    optimized_teardrop_DNA, optimized_teardrop_DNA_5 = [], []
    optimized_linear_DNA, optimized_linear_DNA_5 = [], []
    
    # Loop through files
    for file_name in file_list:
        
        # Safety: Set tip shape function to None for each image
        height_dict, height_dict_updated = None, None
        
        # Set segment length back at 2.5 for each iteration
        improvement_parameters.update({'segment_length': 2.5})
        print('Start new image. \n Segment Length:' + str(improvement_parameters['segment_length']))

        # Import image
        img_original, img_meta_data = import_data.import_ascii(file_path=file_name)

        # Get the imported image in various filtered forms
        img_dict = image_processing.process_img(img_original, mol_min_area=anal_pars['mol_min_area'],
                                                thresh_method=anal_pars['thresh_method'], smooth_method=anal_pars['filter_method'])
        
        # Find the molecules in the image, each molecule is stored as an entry in the molecules list
        molecules = image_processing.get_molecules(img_dict, img_meta_data)
        
        # Create an AFM molecule instance for each individual molecule
        afm_molecules = [cat.AFMMolecule(mol, img_meta_data, anal_pars) for mol in molecules]

        # Split the AFM molecules into lists depending on their type
        mol_bare_DNA = [mol for mol in afm_molecules if mol.mol_pars['type'] == 'Bare DNA']
        mol_teardrop_DNA = [mol for mol in afm_molecules if mol.mol_pars['type'] == 'Teardrop DNA']
        mol_dimer_DNA = [mol for mol in afm_molecules if mol.mol_pars['type'] == 'Potential Dimer']
        mol_pot_lin_DNA = [mol for mol in afm_molecules if mol.mol_pars['type'] == 'Potential Bare DNA']
        mol_pot_td_DNA = [mol for mol in afm_molecules if mol.mol_pars['type'] == 'Potential Teardrop DNA']
    
        print('\nMolecules found:')
        print('{} linear DNA strands'.format(len(mol_bare_DNA)))
        print('{} Teardrop DNA strands'.format(len(mol_teardrop_DNA)))
        print('{} Dimer DNA strands'.format(len(mol_dimer_DNA)))
        print('{} Potential Linear DNA strands'.format(len(mol_pot_lin_DNA)))
        print('{} Potential Teardrop DNA strands'.format(len(mol_pot_td_DNA)))
 
        # Order DNA strands
        linear_DNA = [analysis_linear_DNA.LinearDNA(mol, improvement_parameters) for mol in mol_bare_DNA]
   
        # Get distribution function from Linear DNA - 100 elements suffice
        height_dict = hep.height_distribution_pipeline(linear_DNA[:100], improvement_parameters['max_radius'])
        
        # Refine Linear DNA to update distribution function
        refined_linear_DNA = [analysis_linear_DNA.LinearDNA(mol,
                                improvement_parameters, height_dict) for mol in linear_DNA[:100]]
        
        # Update distribution function
        height_dict_updated = hep.height_distribution_pipeline(refined_linear_DNA, improvement_parameters['max_radius'])
        print('Distribution function established.')
        
        ######
        # Optimize for 2.5 nm
        ######

        # Refine Linear DNA with updated distribution function and desired segment length
        refined_linear_DNA = [analysis_linear_DNA.LinearDNA(mol,
                                improvement_parameters, height_dict_updated) for mol in linear_DNA]
        print('Linear refinement done for 2.5nm')
     
        # Refine Teardrop DNA
        refined_teardrop_DNA = [analysis_teardrop_DNA.TeardropDNA(mol, 
                                improvement_parameters, height_dict_updated) for mol in mol_teardrop_DNA]
        print('Teardrop refinement done for 2.5nm')
        
        #####
        # Optimize for 5 nm
        #####
        
        # Update desired segment length
        improvement_parameters.update({'segment_length': 5})
        print('Updated segment length: ' + str(improvement_parameters['segment_length']))
        
        # Refine Linear DNA with updated height dist. function and desired segment length
        refined_linear_DNA_5 = [analysis_linear_DNA.LinearDNA(mol,
                                improvement_parameters, height_dict_updated) for mol in linear_DNA]
        print('Linear refinement done for 5 nm.')
     
        # Refine Teardrop DNA
        refined_teardrop_DNA_5 = [analysis_teardrop_DNA.TeardropDNA(mol, 
                                improvement_parameters, height_dict_updated) for mol in mol_teardrop_DNA]
        print('Teardrop refinement done for 5 nm.')
        
        # Add to total list
        optimized_teardrop_DNA += [(refined_teardrop_DNA)]
        optimized_teardrop_DNA_5 += [(refined_teardrop_DNA_5)]
        optimized_linear_DNA += [(refined_linear_DNA)]
        optimized_linear_DNA_5 += [(refined_linear_DNA_5)]
        
        print('File done')
        
    return optimized_teardrop_DNA, optimized_teardrop_DNA_5, optimized_linear_DNA, optimized_linear_DNA_5
    
# RUN CODE !
optimized_teardrop_DNA__250_ar, optimized_teardrop_DNA_5__250_ar, optimized_linear_DNA_250_ar, optimized_linear_DNA_5_250_ar = process_pipeline(anal_pars, improvement_parameters, file_list)
#%% Get properties

def get_properties(mol_list, kind='linear'):
    
    # Initialize result lists
    results = {'length_contour': [],
               'length_etoe': [],
               'angles': [],
               'angles_locs': [],
               'angle_exit': [],
               'angle_exit_strep': []
               }
    
    # Image loop
    for this_mol_list in mol_list:
        
        # Get length
        this_mol_length = [mol.mol_pars['length_total'] for mol in this_mol_list]
        results['length_contour'] += this_mol_length
        
        # Get bend angles
        this_mol_angles_locs = [mol.mol_pars['angle_location'] for mol in this_mol_list]
        this_mol_angles = [mol.mol_pars['angle_value'] for mol in this_mol_list]
        results['angles'] += this_mol_angles
        results['angles_locs'] += this_mol_angles_locs
        
        if kind == 'linear':
            this_mol_length_etoe = [mol.mol_pars['length_etoe'] for mol in this_mol_list]
            results['length_etoe'] += this_mol_length_etoe
        
        if kind == 'teardrop':
            this_mol_ea_strep = [mol.mol_pars['exit_angle_strep'] for mol in this_mol_list]
            this_mol_ea_seg = [mol.mol_pars['exit_angle_segment'] for mol in this_mol_list]
            results['angle_exit'] += this_mol_ea_seg
            results['angle_exit_strep'] += this_mol_ea_strep
    
    return results

#%% Some plots

# Get results
this_results = get_properties(optimized_linear_DNA, kind='linear')
datos = this_results['length_contour']
this_results_td = get_properties(optimized_teardrop_DNA, kind='teardrop')
datos_2 = this_results_td['length_contour']

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(5, 5), dpi=400, sharex=True, sharey=True)

# Best fit of data
(mu, sigma) = norm.fit(datos)
(mu_td, sigma_td) = norm.fit(datos_2)

# Update fit
(mu_2, sigma_2) = norm.fit(datos)

# The histogram of the data
n, bins, patches = ax1.hist(datos, bins=20, density=True, facecolor='blue', edgecolor='k', alpha=0.75)
n2, bin2, patches2 = ax2.hist(datos_2, bins=7, density=True, facecolor='blue', edgecolor='k', alpha=0.75)

# Add a 'best fit' line
xmin, xmax = ax1.set_xlim()
x = np.linspace(xmin, xmax, num=100)
y = norm.pdf(x, mu_2, sigma_2)
y_td = norm.pdf(x, mu_td, sigma_td)
ax1.plot(x, y, 'k', linewidth=2)
ax2.plot(x, y_td, 'k', linewidth=2)

# Plot
ax2.set_xlabel('Distance [nm].')
ax1.text(0.6, 0.7, '2143 Linear strands\n\nMean Length: ' + str(np.round(mu_2, 2)) + ' nm\n\nStandard Deviation: ' + str(
    np.round(sigma_2, 2)) + ' nm', c='k', transform=ax1.transAxes, fontsize=7)
ax2.text(0.6, 0.7, '97 Teardrop strands \n\nMean Length: ' + str(np.round(mu_td, 2)) + ' nm\n\nStandard Deviation: ' + str(
    np.round(sigma_td, 2)) + ' nm', c='k', transform=ax2.transAxes, fontsize=7)
ax1.set_ylabel('Rel. Count', fontweight='bold')
ax2.set_ylabel('Rel. Count', fontweight='bold')
fig.tight_layout()
ax1.grid(False)

plt.show()
#%% Plot lenghts 500

# Get results
this_results = get_properties(optimized_linear_DNA_500, kind='linear')
datos = this_results['length_contour']
this_results_td = get_properties(optimized_teardrop_DNA_500, kind='teardrop')
datos_2 = this_results_td['length_contour']

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(5, 5), dpi=400, sharex=True, sharey=True)

# Best fit of data
(mu, sigma) = norm.fit(datos)
(mu_td, sigma_td) = norm.fit(datos_2)


# Update fit
(mu_2, sigma_2) = norm.fit(datos)

# The histogram of the data
n, bins, patches = ax1.hist(datos, bins=7, density=True, facecolor='blue', edgecolor='k', alpha=0.75)
n2, bin2, patches2 = ax2.hist(datos_2, bins=20, density=True, facecolor='blue', edgecolor='k', alpha=0.75)

# Add a 'best fit' line
xmin, xmax = ax1.set_xlim()
x = np.linspace(xmin, xmax, num=100)
y = norm.pdf(x, mu_2, sigma_2)
y_td = norm.pdf(x, mu_td, sigma_td)
ax1.plot(x, y, 'k', linewidth=2)
ax2.plot(x, y_td, 'k', linewidth=2)

# Plot
ax2.set_xlabel('Distance [nm].')
ax1.text(0.6, 0.7, '104 Linear strands\n\nMean Length: ' + str(np.round(mu_2, 2)) + ' nm\n\nStandard Deviation: ' + str(
    np.round(sigma_2, 2)) + ' nm', c='k', transform=ax1.transAxes, fontsize=7)
ax2.text(0.6, 0.7, '387 Teardrop strands \n\nMean Length: ' + str(np.round(mu_td, 2)) + ' nm\n\nStandard Deviation: ' + str(
    np.round(sigma_td, 2)) + ' nm', c='k', transform=ax2.transAxes, fontsize=7)
ax1.set_ylabel('Rel. Count', fontweight='bold')
ax2.set_ylabel('Rel. Count', fontweight='bold')
fig.tight_layout()
ax1.grid(False)

plt.show()

#%% Plot Exit angle

# Get results
this_results_td_500 = get_properties(optimized_teardrop_DNA_5_ar, kind='teardrop')
ea_500 = this_results_td_500['angle_exit']
this_results_td_250 = get_properties(optimized_teardrop_DNA__250_ar, kind='teardrop')
ea_250 = this_results_td_250['angle_exit']

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(5, 5), dpi=400, sharex=True, sharey=True)

# Best fit of data
(mu, sigma) = norm.fit(ea_500)
(mu_2, sigma_2) = norm.fit(ea_250)


# Update fit
#(mu_2, sigma_2) = norm.fit(datos)

# The histogram of the data
n, bins, patches = ax1.hist(ea_500, bins=7, density=True, facecolor='blue', edgecolor='k', alpha=0.75)
n2, bin2, patches2 = ax2.hist(ea_250, bins=7, density=True, facecolor='blue', edgecolor='k', alpha=0.75)

# Add a 'best fit' line
xmin, xmax = ax1.set_xlim()
x = np.linspace(xmin, xmax, num=100)
y = norm.pdf(x, mu, sigma)
y_td = norm.pdf(x, mu_2, sigma_2)
ax1.plot(x, y, 'k', linewidth=2)
ax2.plot(x, y_td, 'k', linewidth=2)

# Plot
ax2.set_xlabel('Exit Angle [deg].')
ax1.text(0.6, 0.7, '387 Teardrop strands\n\nMean Exit Angle: ' + str(np.round(mu, 2)) + ' deg\n\nStandard Deviation: ' + str(
    np.round(sigma, 2)) + ' deg', c='k', transform=ax1.transAxes, fontsize=7)
ax2.text(0.6, 0.7, '107 Teardrop strands \n\nMean Exit Angle: ' + str(np.round(mu_2, 2)) + ' nm\n\nStandard Deviation: ' + str(
    np.round(sigma_2, 2)) + ' nm', c='k', transform=ax2.transAxes, fontsize=7)
ax1.set_ylabel('Rel. Count', fontweight='bold')
ax2.set_ylabel('Rel. Count', fontweight='bold')
fig.tight_layout()
ax1.grid(False)

plt.show()



#%% STUFF BELOW HERE IS OUTDATED
#%%
import itertools

for i in range(1, 5):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=400)
    ax.set_title('Mean angle vs. normalized length, w/ seg_number' + str(i))
    num = len(refined_teardrop_DNA)
    
    improvement_parameters.update({'segment_number': i})
    sum_to_date = np.zeros((1, 1))
    sum_to_date_2 = []
    
    max_length = 0
    for this_td in refined_teardrop_DNA:
        this_td.get_angles(improvement_parameters)
    
        this_td_angles = this_td.mol_pars['angle_distribution']
        this_td_length = this_td.mol_pars['length_contour']
        
        #
        
        #        
        my_length = len(this_td_length)
        if my_length > max_length:
            max_length = my_length
        
        to_add = np.asarray(this_td_angles)
        sum_to_date.resize(to_add.shape)
        
        sum_to_date += to_add
        
        # Version 2:
        sum_to_date_2 = [sum(x) for x in itertools.zip_longest(this_td_angles, sum_to_date_2, fillvalue=0)]

    mean_angles = sum_to_date / num
    mean_angles_2 = [x / num for x in sum_to_date_2]
    x_axis = np.linspace(0, 1, num=max_length)
    ax.plot(x_axis, mean_angles_2)




#%% Plot exit angle histogram

# Initialize
exit_angle_250 = []

# Loop through images
for img_num in td_results_500['td_exit_angle']:
    exit_angle_500 += img_num
    
print('Median exit angle:', np.median(exit_angle_500),'\n Mean exit angle: ',
      np.round(np.mean(exit_angle_500), 3), '+-', np.round(np.std(exit_angle_500) / np.sqrt(len(exit_angle_500)), 3))

# Plot the figure         
fig, ax = plt.subplots(dpi=400)
ax.hist(exit_angle_500, bins=40, histtype='bar', edgecolor='k', facecolor='r') 
ax.set_xlim(0, 250)
ax.set_xlabel('Exit angle [def]',fontsize=15)
ax.set_ylabel('Counts',fontsize=15)
ax.set_title('Exit angle histogram')


#%% Plot of exit angle vs curvature

# Loop through images
max_angle_250 = []

for img_num in td_results['td_angles']:
    
    max_angles = []
    # Loop through teardrops in one image
    for td in img_num:
        max_angle = np.max(td[3:-3])
        max_angles += [max_angle]
    
    max_angle_250 += max_angles 

plt.figure()
plt.scatter(np.asarray(max_angle_250), np.array(exit_angle_250))
plt.xlabel('Max curvature', fontsize=15)
plt.ylabel('Exit angle [deg]', fontsize=15)
plt.title('Exit angle vs. curvature')


#%% Plot curvature for Linear

# Initialize for all images
lin_length_contour_250, lin_angle_250 = [], []

# Loop through images
for img_num_l, img_num_a in zip(lin_results['lin_length_contour'], lin_results['lin_angles']):
    
    length_contour, angles = [], []
    # Loop through linear molecules in one image
    for lin_l, lin_a in zip(img_num_l, img_num_a):
        plt.plot(lin_l, lin_a, linewidth=0.6)
        
        # Add to list
        length_contour += list(lin_l)
        angles += list(lin_a)
    
    lin_length_contour_250 += length_contour
    lin_angle_250 += angles
    
lin_coefs = np.polyfit(lin_length_contour_250, lin_angle_250, 2)
print(lin_coefs)
x_axis = np.linspace(0,1,100)
plt.plot(x_axis, lin_coefs[0] * x ** 2 + lin_coefs[1] * x + lin_coefs[2], linewidth=2.0, color='r')
plt.title('Linear curvature')



