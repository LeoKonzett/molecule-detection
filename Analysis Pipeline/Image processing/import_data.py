# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:38:38 2020

@author: Leo
"""
import os
from tkinter.filedialog import askopenfilename
import numpy as np

def get_file_list(dir_name = 'files'):
    """ Returns a list containing the file paths of files in a given directory.
    IMPORTANT NOTE: This requires the AFM images to be in a separate(!) folder.
    
    Returns:
    -------
    file_list: list
        list object containing the desired file paths
        Example: [C:~/images/500bp/save-2020.11.25-15.01.58.690-3.jpk.asc]
    """
    # Initialize file list
    file_list = []
    
    # Iterate over files in the directory
    for filename in os.scandir(dir_name): 
        if filename.is_file(): 
            file_list.append(filename.path)
            
    return file_list

def import_ascii(file_path=None):
    """
    Function to import an ASCII file with AFM surface height measurements and some
    headerlines. Afterwards converts the floats to uint-8 format. 
    Adapted from Sebastian Konrad.

    Parameters:
    -----------
        file_path - str
            If no file_path is given to the function, a window opens to select the
            desired file manually
            
    Returns:
    --------
        img_original: nd-array
            Original image as nd-array with height values as float, and values < -.05
            deleted
        img_meta_data: dict
            file_name - string
                Name of the file that was imported
            x_pixels - int
                Number of pixels in x-direction
            y_pixels - int
                Number of pixels in y-direction
            x_length - float
                Length of the image in x-direction ->  used to calculate the resolution (x_length/x_pixels)
            pixel_size - float
                Size of a pixel in nanometres
                
    """
    
    # Select path manually using tkinter
    if file_path is None:
        file_path = askopenfilename(title='Select AFM image ASCII file', filetypes=(("ASCII files", "*.asc"),))
    
    # Select image name from file path by "/" splitting
    file_name = file_path.split('/')[-1]
    
    # Initialize image
    img = []
    
    # Read each line, discriminate between header line and height value line by checking if the
    # content of the first entry of the line is a digit or not
    with open(file_path, 'r') as f:
        for line in f:
            try:
                first_entry = line.strip().split()[0][-5:]
                meas_par = line.split()[1]

                if first_entry.isdigit() or first_entry[-5:-3] == 'e-' or first_entry[-4:-2] == 'e-':
                    line = line.strip()
                    floats = [float(x) for x in line.split()]
                    img.append(np.asarray(floats))

                # Find the required measurement information
                elif meas_par == 'x-pixels':
                    x_pixels = float(line.split()[-1])

                # Find the required measurement information
                elif meas_par == 'y-pixels':
                    y_pixels = float(line.split()[-1])
                    
                elif meas_par == 'x-length':
                    x_length = float(line.split()[-1])

            except IndexError:
                pass

    if 'x_pixels' not in locals():
        x_pixels = 'unknown'
        print('The amount of x-pixels was not found in the header')

    if 'y_pixels' not in locals():
        y_pixels = 'unknown'
        print('The amount of y-pixels was not found in the header')

    if 'x_length' not in locals():
        x_length = 'unknown'
        print('The size of the image was not found in the header')

    image = np.asarray(img)

    img_meta_data = {'file_name': file_name,
                     'file_path': file_path,
                     'x_pixels': x_pixels,
                     'x_length': x_length,
                     'y_pixels': y_pixels,
                     'pixel_size': x_length/x_pixels}

    return np.asarray(image), img_meta_data

#%% Older variants not used in the new code.

def from_ascii_to_img(file_list, normalize=False):
    '''This function converts an ascii file to an ndarray.'''
    img_list = []
    for file in file_list:
        image = export_asci_file(file)
        if normalize:
            image -= np.percentile(image, 33)
            image /= np.percentile(image, 99.9)
        img_list.append(image)
    return img_list


def export_asci_file(file_path):
    """Return data from .asc files as float64 ndarray"""
    with open(file_path, 'r') as f:
        row_index = 0
        count = 0
        for line in f:
            if 'Start of Data' in line:
                row_index = count
                break
            count += 1
    data = np.loadtxt(file_path, skiprows=row_index)
    return data

