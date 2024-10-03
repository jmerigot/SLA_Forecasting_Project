""""
Created on Friday 06/09/2024 at 11:18

author: @jmerigot
"""

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         Functions                                     | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

import torch
import numpy as np


def standard_scaler(data, indices):
    """
    Standard scaling function applied to each dataset.

    Args
    ----
    data: numpy
        numpy time series dataset (shape : time, *image_shape)
    indices: list
        list of range of indices of the dataset split to seperate training, validation, testing

    Returns
    -------
    scaled_data: numpy
        scaled numpy time series dataset (shape: time, width, height)
    mean, std: int
        mean and std of the scaled dataset
    """
    
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
    """
    Scaling all datasets using the mean and std of the training set. 
    (would only be used if putting model into widespread production, for example)

    Args
    ----
    data: numpy
        numpy time series dataset (shape : time, *image_shape)
    indices: list
        list of range of indices of the dataset split to seperate training, validation, testing
    mean, std: int
        mean and std of the training set. The default is None.}

    Returns
    -------
    scaled_data: numpy
        scaled numpy time series dataset, of shape: time, width, height (e.g.: [10, 128, 128] for sequence of 10 images)
    mean, std: int
        mean and std of the scaled dataset
    """
    
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



def custom_transform(image):
    """
    Custom transformation applied to each image in the datasets.
    Transformation:
     - Transform to tensor,
     - Flip upside down since all data is South-side up,
     - Crop to upper 128x128 pixels to isolate dynamic region of the Gulf Stream.

    Args
    ----
    image: numpy
        singular numpy image from the dataset

    Returns
    -------
    image: tensor
        transformed image as a torch tensor
    """
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    image = torch.flip(image, [1])  
    image = image[:, :128, :128]
    return image



def inverse_scale(image, mean, std):
    """
    Inverse scale function for predicted images before visualization and metric evalution.

    Args
    ----
    image: tensor
        the predicted image as a tensor of shape torch.Size([128, 128])
    mean, std: tensor
        the corresponding mean and std of the image of shape torch.Size([1, 1])

    Returns
    -------
    inverse_image: tensor
        the inversed scaled predicted image ready for visualisation
    """
    
    inverse_image = image * std + mean
    return inverse_image


def inverse_transform(image, mean, std):
    """
    Inverse transform function for predicted images before visualization and metric evalution.
    Inverse scale and flip back to original orientation.

    Args
    ----
    image: tensor
        the predicted image as a tensor of shape torch.Size([128, 128])
    mean, std: tensor
        the corresponding mean and std of the image of shape torch.Size([1, 1])

    Returns
    -------
    inverse_image: tensor
        the inversed transformed predicted image ready for visualisation
    """
    
    inverse_image = image * std + mean
    inverse_image = torch.flipud(inverse_image)
    return inverse_image