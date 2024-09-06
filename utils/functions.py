""""
Created on Friday 06/09/2024 at 11:18

author: @jmerigot
"""

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         Functions                                     | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

import numpy as np


def standard_scaler(data, indices):
    transformed_data = []
    
    for i in indices:
        image = data[i]
        transformed_image = np.flipud(image)
        transformed_image = transformed_image[:128, :128]
        transformed_data.append(transformed_image)
    
    transformed_data = np.array(transformed_data)
    
    mean = np.mean(transformed_data, axis=(1, 2), keepdims=True)
    std = np.std(transformed_data, axis=(1, 2), keepdims=True)
    std[std == 0] = 1
    scaled_data = (transformed_data - mean) / std
    
    return scaled_data, mean, std

def standard_scaler_with_train(data, indices, mean=None, std=None):
    transformed_data = []
    
    for i in indices:
        image = data[i]
        transformed_image = np.flipud(image)
        transformed_image = transformed_image[:128, :128]
        transformed_data.append(transformed_image)
    
    transformed_data = np.array(transformed_data)
    
    # if mean and std not provided, calculate them from training data
    if mean is None or std is None:
        mean = np.mean(transformed_data, axis=(1, 2), keepdims=True)
        std = np.std(transformed_data, axis=(1, 2), keepdims=True)
        std[std == 0] = 1
    
    # allow for proper broadcasting of shapes
    mean = mean[:transformed_data.shape[0]]
    std = std[:transformed_data.shape[0]]
    
    scaled_data = (transformed_data - mean) / std
    
    return scaled_data, mean, std