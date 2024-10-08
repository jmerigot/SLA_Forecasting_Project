""""
Created on Friday 06/09/2024 at 11:40

author: @jmerigot
"""

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         TRAINER                                       | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping

from config import dataloader_config, model_name, models_dict, model_params
from LightningModule import SLA_Lightning_Module
from Dataloaders import time_series_module


"""
This is the file that should be run to train the model.
We use a Lightining Trainer, for more info see: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html
"""


"""Define EarlyStopping callback for training.
"""
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.000,
    patience=8,
    verbose=True,
    mode='min'  # mode can be 'min', 'max', or 'auto'. 'min' means training will stop when the quantity monitored has stopped decreasing
)  

"""Define Lightning Trainer.
"""
trainer = L.Trainer(
    max_epochs=200,
    callbacks=[early_stopping_callback],
    accelerator="auto",
    devices=[0],  # Use GPU if available, you can specify more explicitly, e.g., `devices=1` if one GPU is to be used.
    log_every_n_steps=10,
)


#---------------------------------------------#
#                  TRAINING                   #
#---------------------------------------------#

# define model using LightningModule
model = SLA_Lightning_Module(models_dict, model_name, **model_params)

# load data
data_module = time_series_module(dataloader_config)
data_module.setup()

# prepate datasets
train_set, train_loader = data_module.train_dataloader()
val_set, val_loader = data_module.val_dataloader()
test_set, test_loader, test_mean, test_std = data_module.test_dataloader()

# launch training
model_training = trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
print("Training completed!")


# validate training
model_validation = trainer.validate(model, dataloaders=val_loader)

# testing results
model_testing = trainer.test(model, dataloaders=test_loader)

# make predictions
predictions = trainer.predict(model, dataloaders=test_loader)