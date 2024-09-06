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
from models.SLA_SmaAt_UNet import SmaAt_UNet_SLA
from models.SLA_SST_SmaAt_UNet import SmaAt_UNet_SLA_SST


"""
Only modify this file to make changes to forecasting process.

 - sequence_length variable determines the prediction horizon
 
 - model_name determines the model being used 
   (MAKE SURE to change include_sst to False in dataloader_config if using SLA-SmaAt-UNet)
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


models_dict = {
    "smaat_unet_sla": SmaAt_UNet_SLA,
    "smaat_unet_sla_sst": SmaAt_UNet_SLA_SST
}

#################################
model_name = 'smaat_unet_sla_sst'
#################################

if model_name == 'smaat_unet_sla':
    model_params = {'n_channels': sequence_length, 'n_classes': sequence_length}
elif model_name == 'smaat_unet_sla_sst':
    model_params = {'n_channels': sequence_length*2, 'n_classes': sequence_length*2}