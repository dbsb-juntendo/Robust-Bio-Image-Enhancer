# -*- coding: utf-8 -*-
"""
Author: Yuichi Wada Ph.D. 
The Department of Biochemistry II Juntendo University Graduate School of Medicine

RobustBioImageEnhancer.py
Copyright (C) 2024 Yuichi Wada

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

PATHNAME =  "Your Pathname"
Imagefiletype =".tif"

# Get file list
input_file_names = glob.glob(PATHNAME + "/**/*"+Imagefiletype, recursive=True)

for file_name in input_file_names:
    # Read image
    image = io.imread(file_name)
    
    # Compute histogram
    histogram, bin_edges = np.histogram(image, bins=256)
    
    # Define lower threshold
    min_bin_index = np.argmin(histogram[1:12])
    lower_threshold = bin_edges[min_bin_index - 1].astype(np.int16)
    
    alpha= np.median(image[image>lower_threshold])/DESIRED_MEDIAN
    alpha = round(alpha, 3).astype(np.float16)
    
    # Scale intensities and adjust contrast
    image = image.astype(np.float32)
    image[image > lower_threshold] = image[image > lower_threshold] * alpha
    
    # Compute brightness adjustment factor
    beta = DESIRED_MEDIAN - np.median(image)
    beta = round(beta)
    
    # Adjust brightness and clip values
    image[image > lower_threshold] = image[image > lower_threshold] + beta
    image = np.clip(image, 0, 65535).astype(np.uint16)
    
    # Save the adjusted image
    output_file_name = file_name.replace(Imagefiletype, "_adjusted.tif")
    tif.imwrite(output_file_name, image)
