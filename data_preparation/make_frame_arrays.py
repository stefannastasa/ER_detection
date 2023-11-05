"""
Create and save clips as numpy arrays using separated frame images.
(used for 3_corner_point_annotation.py and 4_retrieve_physical_scale.py)
"""

import os
import pickle
import sys

sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))

import numpy as np
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from imageio import imread
from natsort import natsorted
from skimage.util import img_as_float

from utils.config import images_folder, arrays_folder, info_folder

# define directories
input_folder = images_folder
output_folder = arrays_folder
# get the dictionary with original clip shapes

skipped_cases = []

# --------------------------------------------------------------------------------

# check if output folder exists
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

# get all files in the input folder
cases = os.listdir(input_folder)
shape_dict = 'shape_dictionary.pkl'
frame_cases = {}
for case in cases:
    output_case_folder = os.path.join(output_folder, case)
    images_array = []
    images = os.listdir(os.path.join(input_folder, case))

    if not os.path.isdir(output_case_folder):
        os.mkdir(output_case_folder)

    for image in tqdm(images):
        image_data = imread(os.path.join(input_folder, case, image))
        image_shape = image_data.shape
        image_array = np.zeros((1, *image_shape), dtype=np.float16)
        image_array[0,...] = img_as_float(image_data)

        np.save(os.path.join(output_case_folder, image), image_array)


