""""
Created on Friday 06/09/2024 at 11:29

author: @jmerigot
"""

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         CONFIG                                        | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

from torchvision import transforms
from utils.functions import custom_transform

"""
Only modify this file to make changes to forecasting process.
sequence_length variable determines the prediction horizon.
"""


transform = transforms.Compose([
    transforms.Lambda(custom_transform)
])

####################
sequence_length = 10
####################

dataloader_config={'dataset_path': ['/data/home/jmerigot/start_data/sla_ts.npy', '/data/home/jmerigot/start_data/sst_ts.npy'],
                   'include_sst': True, # set to False if using the SLA-SmaAt-UNet without SST data
                   'length_source': sequence_length, # how many timestep in inputs
                   'length_target': sequence_length, # how many timestep for prediction (if None monitered with prediction probabilities)
                   'timestep': 1, # days between each inputs
                   'transform': transform,
                   'valid_ratio': 0.10, # ratio to use for validation and test sets
                   'batch_size': 16,
                   'small_train': False,
                   'model_is_3dconv': False, # in case using a model with 3D convolutions; rare
                   'scale_with_train': False} # in case want to standard scale all datasets using the mean/std of the training set; rare