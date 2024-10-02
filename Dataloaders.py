""""
Created on Friday 06/09/2024 at 11:14

author: @jmerigot
"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         DATALOADERS                                   | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

import torch
import lightning as L
import numpy as np

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler
from utils.functions import standard_scaler, standard_scaler_with_train


"""
Custom Dataset using LightningDataModules and inherited from Pytorch Dataset
Take .npy image time series as dataset and create time series for training of length L_source+L_target
"""


class time_series_dataset(Dataset):

    def __init__(self, datasets, L_source, L_target, dt, transform=None, model_is_3dconv=False, include_sst=False):
        """
        Parameters
        ----------
        dataset : numpy
            numpy time series dataset (shape : time, *image_shape)
        L_source, L_target : int
            lengths of network input source and output target
        dt : int
            timestep for the time series in days
        transform : pytorch transform, optional
            transformation to apply to the data. The default is None.
        """
        self.include_sst = include_sst
        if self.include_sst:
            self.dataset_sla = datasets[0]
            self.dataset_sst = datasets[1]
        else:    
            self.dataset_sla = datasets
            
        self.transform = transform
        self.L_source = L_source * dt
        self.L_target = L_target * dt
        self.dt = dt
        self.model_is_3dconv = model_is_3dconv


    def __len__(self):
        return len(self.dataset_sla) - self.L_source - self.L_target + 1
    
    def __getitem__(self, idx):
        """
        Main function of the CustomDataset class. 
        """
        input_start = idx
        input_end = idx + self.L_source
        output_end = input_end + self.L_target
        
        if self.include_sst:
            item = {}
            item['input_sla'], item['input_sst'] = self.dataset_sla[input_start:input_end], self.dataset_sst[input_start:input_end]
            item['target_sla'], item['target_sst'] = self.dataset_sla[input_end:output_end], self.dataset_sst[input_end:output_end]
        else:
            input_images = self.dataset_sla[input_start:input_end]
            target_images = self.dataset_sla[input_end:output_end]
        
        if self.transform:
            if self.include_sst:
                item['input_sla'] = torch.stack([self.transform(image) for image in item['input_sla']])
                item['input_sst'] = torch.stack([self.transform(image) for image in item['input_sst']])
                item['target_sla'] = torch.stack([self.transform(image) for image in item['target_sla']])
                item['target_sst'] = torch.stack([self.transform(image) for image in item['target_sst']])
            else:
                input_images = torch.stack([self.transform(image) for image in input_images])
                target_images = torch.stack([self.transform(image) for image in target_images])
        
        # if using a model with 3d convolutions, reorganize dimensions
        if self.model_is_3dconv is True:
            if self.include_sst:
                item['input_sla'], item['input_sst'] = item['input_sla'].permute(1, 0, 2, 3), item['input_sst'].permute(1, 0, 2, 3)
                item['target_sla'], item['target_sst'] = item['target_sla'].permute(1, 0, 2, 3), item['target_sst'].permute(1, 0, 2, 3)
            else:
                input_images = input_images.permute(1, 0, 2, 3)
                target_images = target_images.permute(1, 0, 2, 3)
        # else, reshape data to flatten the time channel and sequence dimensions into one dimension for 2d convolutions
        else:
            if self.include_sst:
                item['input_sla'], item['input_sst'] = item['input_sla'].view(-1, 128, 128), item['input_sst'].view(-1, 128, 128)
                item['target_sla'], item['target_sst'] = item['target_sla'].view(-1, 128, 128), item['target_sst'].view(-1, 128, 128)
            else:
                input_images = input_images.view(-1, 128, 128)
                target_images = target_images.view(-1, 128, 128)

        if self.include_sst:
            return item
        else:
            return input_images, target_images

def load_item(index, dataset, L_source, L_target, dt):
    """
    Parameters
    ----------
    index : int
    dataset : numpy
        numpy time series dataset (shape : time, *image_shape)
    L_source, L_target : int
        length in days of network input source and output target
    dt : int
        timestep for the time series in days

    Returns
    -------
    source : numpy
        time series of shape (L_source/dt, *image_shape).
    target : numpy
        time series of shape (L_target/dt, *image_shape).
    """
    
    index_source=[i for i in range(index, index+L_source, dt)]
    index_target=[i for i in range(index+L_source, index+L_source+L_target, dt)]
    
    source = np.stack([dataset[i] for i in index_source])
    target = np.stack([dataset[i] for i in index_target])
    
    return source, target


class time_series_module(L.LightningDataModule):
    
    def __init__(self, dataloader_config):
        super().__init__()
        """
        Parameters
        ----------
        dataloader_config : dict
            dataloader configuration (see config.py).

        Returns
        -------
        Pytorch Dataloader or dict of Dataloader 
        """
        
        self.dataloader_config = dataloader_config

        if self.dataloader_config['include_sst']:
            self.dataset = [np.load(i) for i in self.dataloader_config['dataset_path']]
        else:
            self.dataset = np.load(self.dataloader_config['dataset_path'])
    
    def setup(self):
        # set percentage for test and validation sets
        test_ratio = self.dataloader_config['valid_ratio']  # 10% for testing
        valid_ratio = self.dataloader_config['valid_ratio']  # 10% for validation
        if self.dataloader_config['include_sst']:
            total_images = len(self.dataset[0])
        else:
            total_images = len(self.dataset)

        # calculate split indices
        test_size = int(total_images * test_ratio)
        valid_size = int(total_images * valid_ratio)
        train_size = total_images - test_size - valid_size
        
        # make datasets for training, validation, and testing
        if self.dataloader_config['include_sst']:
            
            train_set = (self.dataset[0][test_size:train_size+test_size], self.dataset[1][test_size:train_size+test_size])
            val_set = (self.dataset[0][train_size+test_size:total_images], self.dataset[1][train_size+test_size:total_images])
            test_set = (self.dataset[0][0:test_size], self.dataset[1][0:test_size])
            
            # for testing specific cutoffs
            """
            train_set = (self.dataset[0][365:7776], self.dataset[1][365:7776])
            val_set = (self.dataset[0][7777:9051], self.dataset[1][7777:9051])
            test_set = (np.concatenate((self.dataset[0][0:364], self.dataset[0][9052:9860]), axis=0), 
                        np.concatenate((self.dataset[1][0:364], self.dataset[1][9052:9860]), axis=0))
            """
        else:
            
            train_set = self.dataset[test_size:train_size + test_size]
            val_set = self.dataset[train_size + test_size:total_images]
            test_set = self.dataset[0:test_size]
            
            # for testing specific cutoffs
            """
            train_set = self.dataset[365:7776]
            val_set = self.dataset[7777:9051]
            test_set = np.concatenate((self.dataset[0:364], self.dataset[9052:9860]), axis=0)
            """
        
        # standard scaling of the datasets
        if self.dataloader_config['scale_with_train']:
            if self.dataloader_config['include_sst']:
                scaled_train_set_sla, train_mean_sla, train_std_sla = standard_scaler_with_train(train_set[0], list(range(len(train_set[0]))))
                scaled_val_set_sla, _, _ = standard_scaler_with_train(val_set[0], list(range(len(val_set[0]))), 
                                                                      mean=train_mean_sla, std=train_std_sla)
                scaled_test_set_sla, self.mean_sla, self.std_sla = standard_scaler_with_train(test_set[0], list(range(len(test_set[0]))), 
                                                                                              mean=train_mean_sla, std=train_std_sla)
                
                scaled_train_set_sst, train_mean_sst, train_std_sst = standard_scaler_with_train(train_set[1], list(range(len(train_set[1]))))
                scaled_val_set_sst, _, _ = standard_scaler_with_train(val_set[1], list(range(len(val_set[1]))), 
                                                                      mean=train_mean_sst, std=train_std_sst)
                scaled_test_set_sst, self.mean_sst, self.std_sst = standard_scaler_with_train(test_set[1], list(range(len(test_set[1]))), 
                                                                                              mean=train_mean_sst, std=train_std_sst)
                scaled_train_set = (scaled_train_set_sla, scaled_train_set_sst)
                scaled_val_set = (scaled_val_set_sla, scaled_val_set_sst)
                scaled_test_set = (scaled_test_set_sla, scaled_test_set_sst)
                self.mean = (self.mean_sla, self.mean_sst)
                self.std = (self.std_sla, self.std_sst)
            else:
                scaled_train_set, train_mean, train_std = standard_scaler_with_train(train_set, list(range(len(train_set))))
                scaled_val_set, _, _ = standard_scaler_with_train(val_set, list(range(len(val_set))),
                                                                  mean=train_mean, std=train_std)
                scaled_test_set, self.mean, self.std = standard_scaler_with_train(test_set, list(range(len(test_set))), 
                                                                                  mean=train_mean, std=train_std)
        else:
            if self.dataloader_config['include_sst']:
                scaled_train_set_sla, train_mean_sla, train_std_sla = standard_scaler(train_set[0], list(range(len(train_set[0]))))
                scaled_val_set_sla, _, _ = standard_scaler(val_set[0], list(range(len(val_set[0]))))
                scaled_test_set_sla, self.mean_sla, self.std_sla = standard_scaler(test_set[0], list(range(len(test_set[0]))))
                
                scaled_train_set_sst, train_mean_sst, train_std_sst = standard_scaler(train_set[1], list(range(len(train_set[1]))))
                scaled_val_set_sst, _, _ = standard_scaler(val_set[1], list(range(len(val_set[1]))))
                scaled_test_set_sst, self.mean_sst, self.std_sst = standard_scaler(test_set[1], list(range(len(test_set[1]))))
                
                scaled_train_set = (scaled_train_set_sla, scaled_train_set_sst)
                scaled_val_set = (scaled_val_set_sla, scaled_val_set_sst)
                scaled_test_set = (scaled_test_set_sla, scaled_test_set_sst)
                self.mean = (self.mean_sla, self.mean_sst)
                self.std = (self.std_sla, self.std_sst)
            else:
                scaled_train_set, _, _ = standard_scaler(train_set, list(range(len(train_set))))
                scaled_val_set, _, _ = standard_scaler(val_set, list(range(len(val_set))))
                scaled_test_set, self.mean, self.std = standard_scaler(test_set, list(range(len(test_set))))
        
        # create dataset objects        
        self.training_set = time_series_dataset(scaled_train_set,
                                                self.dataloader_config['length_source'],
                                                self.dataloader_config['length_target'],
                                                self.dataloader_config['timestep'],
                                                self.dataloader_config['transform'],
                                                self.dataloader_config['model_is_3dconv'],
                                                self.dataloader_config['include_sst'])
        
        self.validation_set = time_series_dataset(scaled_val_set, 
                                                  self.dataloader_config['length_source'],
                                                  self.dataloader_config['length_target'],
                                                  self.dataloader_config['timestep'],
                                                  self.dataloader_config['transform'],
                                                  self.dataloader_config['model_is_3dconv'],
                                                  self.dataloader_config['include_sst'])
        
        self.testing_set = time_series_dataset(scaled_test_set,
                                               self.dataloader_config['length_source'],
                                               self.dataloader_config['length_target'],
                                               self.dataloader_config['timestep'],
                                               self.dataloader_config['transform'],
                                               self.dataloader_config['model_is_3dconv'],
                                               self.dataloader_config['include_sst'])
            
    def train_dataloader(self):
        training_generator = DataLoader(self.training_set,
                                        num_workers=15, 
                                        batch_size=self.dataloader_config['batch_size'], 
                                        shuffle=True)
        
        if self.dataloader_config['small_train']:
            train_sampler = SubsetRandomSampler([i for i in range(0, len(self.training_set), 2)]) #only half data to compute faster 
            training_generator = DataLoader(self.training_set,
                                            batch_size=self.dataloader_config['batch_size'],
                                            sampler=train_sampler)
        
        return self.training_set, training_generator
    
    def val_dataloader(self):
        validation_generator = DataLoader(self.validation_set,
                                          num_workers=15,
                                          batch_size=self.dataloader_config['batch_size'],
                                          shuffle=False,
                                          persistent_workers=False)
        
        return self.validation_set, validation_generator
    
    def test_dataloader(self):
        sampler = SequentialSampler(self.testing_set)
        
        testing_generator = DataLoader(self.testing_set,
                                       num_workers=15,
                                       batch_size=1,#self.dataloader_config['batch_size'],
                                       shuffle=False,
                                       sampler=sampler)
        
        return self.testing_set, testing_generator, self.mean, self.std