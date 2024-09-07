""""
Created on Friday 07/09/2024 at 18:20

author: @jmerigot
"""

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                   Plot SLA Sequence                                   | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from config import model_name
from Trainer import test_mean, test_std, test_set, predictions

"""
Run this file to plot visualizations of forecasted SLA sequences.
"""


# load max and min values for norm of plot
original_data_sla = np.load('/data/home/jmerigot/start_data/sla_ts.npy')
max_value_sla = np.max(original_data_sla)
min_value_sla = np.min(original_data_sla)

# create a single norm to be shared across all images
norm = colors.Normalize(vmin=min_value_sla*100, vmax=max_value_sla*100)

# prepare prediction images
use_sst = False
if model_name == "smaat_unet_sla":
    all_predictions = torch.cat([batch.cpu() for batch in predictions], dim=0)
elif model_name == "smaat_unet_sst":
    use_sst = True
    all_predictions = torch.cat(predictions, dim=0)
elif model_name == "smaat_unet_sla_sst":
    use_sst = True
    all_sla = []
    for i in range(len(predictions)):
        all_sla.append(predictions[i][0])
    all_predictions = torch.cat(all_sla, dim=0)
else:
    raise ValueError("Model name not defined properly.")
print(f"Predictions Tensor Shape: {all_predictions.shape}")

# convert mean and std to tensors
mean = torch.tensor(test_mean, dtype=torch.float32)
std = torch.tensor(test_std, dtype=torch.float32)
mean = mean[0]
std = std[0]

# inverse scale function for images
def inverse_scale(image, mean, std):
    return image * std + mean

# function to visualize sequences of predictions and targets side by side
def visualize_prediction_and_target_sequence(predictions, targets, mean=None, std=None, sample_index=0, use_sst=False):
    num_time_steps = predictions.shape[0]
    fig, axs = plt.subplots(3, num_time_steps, figsize=(5 * num_time_steps, 15))  # 3 rows, num_time_steps columns

    for time_step in range(num_time_steps):
        prediction = predictions[time_step]
        target = targets[time_step]

        if mean is not None and std is not None:
            if use_sst == True:
                prediction = inverse_scale(prediction, mean[time_step], std[time_step])
                target = inverse_scale(target, mean[time_step], std[time_step])
            else:
                prediction = inverse_scale(prediction, mean, std)
                target = inverse_scale(target, mean, std)
        
        prediction = torch.flipud(prediction)
        target = torch.flipud(target)

        im1 = axs[0, time_step].imshow(prediction.squeeze()*100, cmap='seismic', norm=norm)
        axs[0, time_step].set_title(f'Predicted Image for t = {sample_index}, step = {time_step+1}')
        axs[0, time_step].axis('on')
        axs[0, time_step].text(0.5, -0.15, f't+{time_step+1}\n', ha='center', va='center', transform=axs[0, time_step].transAxes)

        im2 = axs[1, time_step].imshow(target.squeeze()*100, cmap='seismic', norm=norm)
        axs[1, time_step].set_title(f'Target Image for t = {sample_index}, step = {time_step+1}')
        axs[1, time_step].axis('on')
        
        im3 = axs[2, time_step].imshow((target.squeeze()-prediction.squeeze())*100, cmap='seismic', norm=norm)
        axs[2, time_step].set_title(f'Difference for t = {sample_index}, step = {time_step+1}')
        axs[2, time_step].axis('on')
        
    # add colorbars
    cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.05, shrink=1.0, aspect=50, pad=0.02)
    cbar.set_label('Anomaly (cm)')

    plt.show()


# choose wich indices to plot
sample_index = 0  # Change this index to visualize different samples
sample = test_set[sample_index]  # Get the first sample

if isinstance(sample, tuple):
    print("Instance of sample: Tuple")
    data, targets = sample
    print(f"Data shape: {data.shape}")
    print(f"Target shape: {targets.shape}")

    # Get the corresponding predictions
    predictions_first_batch = all_predictions[sample_index]

    # Visualize the prediction and target sequences side by side
    visualize_prediction_and_target_sequence(predictions_first_batch, targets, mean, std, sample_index, use_sst)
elif isinstance(sample, dict):
    print("Instance of sample: Dict")
    data, targets = sample['input_sla'], sample['target_sla']
    print(f"Data shape: {data.shape}")
    print(f"Target shape: {targets.shape}")

    # Get the corresponding predictions
    predictions_first_batch = all_predictions[sample_index]

    # Visualize the prediction and target sequences side by side
    visualize_prediction_and_target_sequence(predictions_first_batch, targets, mean, std, sample_index, use_sst)
else:
    print(f"Sample shape: {sample.shape}")