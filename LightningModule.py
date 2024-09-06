""""
Created on Friday 06/09/2024 at 11:35

author: @jmerigot
"""

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                               Main Lightning Module                                   | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

import torch
from torch import nn
import torch.optim as optim
import lightning as L

"""
This is where the main model training process occurs using a Lightning Module.
Here, we do our training step, validation step, testing step, and final predict step.
We also load our models, configure the optimizers, and compute the loss.
"""


class SLA_Lightning_Module(L.LightningModule):
    def __init__(self, models_dict, model_name, **kwargs):
        super().__init__()
        
        self.model_name = model_name
        self.model = models_dict[model_name](**kwargs)
        
        self.loss_criterion = nn.MSELoss(reduction='none')

    def forward(self, x1, x2=None):
        if self.model_name == "smaat_unet_sla":
            return self.model(x1)
        elif self.model_name == "smaat_unet_sla_sst":
            return self.model(x1, x2)
        else:
            raise ValueError("WRONG MODEL NAME INPUT")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.0001, weight_decay=0.01)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                                    factor=0.5, patience=3, 
                                                                    verbose=True),
            'monitor': 'val_loss'
        }
        return [optimizer], [lr_scheduler]
    
    def compute_loss(self, predictions, target_sequence, weighted_loss='quadratic'):
        mse_loss = self.loss_criterion(predictions, target_sequence)
        
        sequence_length = target_sequence.size(1)
        
        if weighted_loss == 'exponential':
            base = 4.0 # increase base to increase weights on later images
            weights = torch.exp(torch.linspace(0, 1, steps=sequence_length) * base).to(target_sequence.device)
        elif weighted_loss == 'quadratic':
            weights = torch.arange(1, sequence_length + 1).float().to(target_sequence.device)
            weights = (weights / sequence_length) ** 2  # square the weights to get a quadratic increase
            
        weights = weights / weights.sum() # normalize to 1
        
        weighted_mse_loss = mse_loss * weights.view(1, sequence_length, 1, 1)
        loss = weighted_mse_loss.mean()
        return loss

    def training_step(self, batch, batch_idx):
        if self.model_name == 'smaat_unet_sla':
            input_sequences, target_sequences = batch
            predictions = self(input_sequences)
            loss = self.compute_loss(predictions, target_sequences)
        else:
            input_sla, input_sst, target_sla, target_sst = batch['input_sla'], batch['input_sst'], batch['target_sla'], batch['target_sst']
            if self.model_name == 'smaat_unet_sla_sst':
                predictions_sla, predictions_sst = self(input_sla, input_sst)
                loss_sla = self.compute_loss(predictions_sla, target_sla)
                loss_sst = self.compute_loss(predictions_sst, target_sst)
                loss = loss_sla + loss_sst  
            elif self.model_name == 'smaat_unet_sst':
                predictions = self(input_sla, input_sst)
                loss = self.compute_loss(predictions, target_sla)
                
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.model_name == 'smaat_unet_sla':
            input_sequences, target_sequences = batch
            predictions = self(input_sequences)
            loss = self.compute_loss(predictions, target_sequences)
        else:
            input_sla, input_sst, target_sla, target_sst = batch['input_sla'], batch['input_sst'], batch['target_sla'], batch['target_sst']
            if self.model_name == 'smaat_unet_sla_sst':
                predictions_sla, predictions_sst = self(input_sla, input_sst)
                loss_sla = self.compute_loss(predictions_sla, target_sla)
                loss_sst = self.compute_loss(predictions_sst, target_sst)
                loss = loss_sla + loss_sst  
            elif self.model_name == 'smaat_unet_sst':
                predictions = self(input_sla, input_sst)
                loss = self.compute_loss(predictions, target_sla)
                
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        if self.model_name == 'smaat_unet_sla':
            input_sequences, target_sequences = batch
            predictions = self(input_sequences)
            loss = self.compute_loss(predictions, target_sequences)
        else:
            input_sla, input_sst, target_sla, target_sst = batch['input_sla'], batch['input_sst'], batch['target_sla'], batch['target_sst']
            if self.model_name == 'smaat_unet_sla_sst':
                predictions_sla, predictions_sst = self(input_sla, input_sst)
                loss_sla = self.compute_loss(predictions_sla, target_sla)
                loss_sst = self.compute_loss(predictions_sst, target_sst)
                loss = loss_sla + loss_sst  
            elif self.model_name == 'smaat_unet_sst':
                predictions = self(input_sla, input_sst)
                loss = self.compute_loss(predictions, target_sla)
                
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        if self.model_name == 'smaat_unet_sla':
            input_sequences, _ = batch
            predictions = self(input_sequences)
        else:
            input_sla, input_sst, _, _ = batch['input_sla'], batch['input_sst'], batch['target_sla'], batch['target_sst']
            if self.model_name == 'smaat_unet_sla_sst':
                predictions_sla, predictions_sst = self(input_sla, input_sst) 
                predictions = (predictions_sla, predictions_sst)
            elif self.model_name == 'smaat_unet_sst':
                predictions = self(input_sla, input_sst)
                
        return predictions