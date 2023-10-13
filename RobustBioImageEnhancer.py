# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 19:57:59 2023
@author: Yuichi Wada Ph.D. 
The Department of Biochemistry and Systems Biomedicine 
at Juntendo University Graduate School of Medicine

RobustBioImageEnhancer.py
Copyright (C) 2023 Yuichi Wada

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""

import numpy as np
import tifffile as tif
from skimage import io
import glob
from scipy import stats

# Constants
MAX_ROI_INTENSITY = round(65535 * 0.5)
EXCLUDE_INTENSITY_THRESHOLD = round(65535 * 0.95)
DESIRED_MEDIAN = 10000
DESIRED_WIDTH = 30000

PATHNAME = "Your Pathname"

# Get file list
input_file_names = glob.glob(PATHNAME + "*.tif")

for file_name in input_file_names:
    # Read image
    image = io.imread(PATHNAME + file_name)
    
    # Compute histogram
    histogram, bin_edges = np.histogram(image, bins=256)
    
    # Define lower threshold
    min_bin_index = np.argmin(histogram[1:12])
    lower_threshold = bin_edges[min_bin_index - 1].astype(np.int16)
    
    # Determine the peak of ROI and bin width
    peak_roi_index = np.argmax(histogram[12:255])
    roi_center_initial = bin_edges[peak_roi_index + 10]
    bin_width = (peak_roi_index + 12) - min_bin_index
    width_ratio = bin_width * 2 * 256
    
    # Suppress high intensity noise
    image[image > EXCLUDE_INTENSITY_THRESHOLD] = 0
    
    # Compute desired standard deviation
    desired_std = (DESIRED_WIDTH * 0.90) / (2 * stats.norm.ppf(0.95))
    desired_std = round(desired_std, 3)
    
    # Compute scaling factor
    alpha = desired_std / width_ratio
    alpha = round(alpha, 3).astype(np.float16)
    
    # Scale intensities and adjust contrast
    image = image.astype(np.float32)
    image[image > lower_threshold] = image[image > lower_threshold] * alpha
    
    # Compute brightness adjustment factor
    roi_center_adjusted = roi_center_initial * alpha
    beta = DESIRED_MEDIAN - roi_center_adjusted - lower_threshold
    beta = round(beta)
    
    # Adjust brightness and clip values
    image[image > lower_threshold] = image[image > lower_threshold] + beta
    image = np.clip(image, 0, 65535).astype(np.uint16)
    
    # Save the adjusted image
    output_file_name = file_name.replace(".tif", "_adjusted.tif")
    tif.imwrite(output_file_name, image)
