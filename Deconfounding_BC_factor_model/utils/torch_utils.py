'''
import torch
from torch.utils.data import Dataset

class dataset_base(Dataset):
    def __init__(self, dataset):
        self.sequence_length = dataset['sequence_length']
        self.previous_covariates = dataset['previous_covariates']
        self.previous_treatments = dataset['previous_treatments']
        self.covariates = dataset['covariates']
        self.treatments = dataset['treatments']
        self.outcomes = dataset['outcomes']

    def __getitem__(self, index):
        previous_covariate = self.previous_covariates[index]
        previous_treatment = self.previous_treatments[index]
        covariate = self.covariates[index]
        treatment = self.treatments[index]
        outcome = self.outcomes[index]
        return torch.from_numpy(previous_covariate), torch.from_numpy(previous_treatment), torch.from_numpy(
            covariate), torch.from_numpy(treatment), torch.from_numpy(outcome)

    def __len__(self):
        return self.previous_covariates.shape[0]

def get_dataset_splits(dataset, train_index, val_index, test_index, use_predicted_confounders):
    if use_predicted_confounders:
        dataset_keys = ['previous_covariates', 'previous_treatments', 'covariates', 'treatments',
                        'predicted_confounders', 'outcomes']
    else:
        dataset_keys = ['previous_covariates', 'previous_treatments', 'covariates', 'treatments', 'outcomes']

    dataset_train = dict()
    dataset_val = dict()
    dataset_test = dict()
    for key in dataset_keys:
        dataset_train[key] = dataset[key][train_index, :, :]
        dataset_val[key] = dataset[key][val_index, :, :]
        dataset_test[key] = dataset[key][test_index, :, :]

    _, length, num_covariates = dataset_train['covariates'].shape

    key = 'sequence_length'
    dataset_train[key] = dataset[key][train_index]
    dataset_val[key] = dataset[key][val_index]
    dataset_test[key] = dataset[key][test_index]

    dataset_map = dict()

    dataset_map['num_time_steps'] = length
    dataset_map['training_data'] = dataset_train
    dataset_map['validation_data'] = dataset_val
    dataset_map['test_data'] = dataset_test

    return dataset_map


import torch
from torch.utils.data import Dataset

class dataset_base(Dataset):
    def __init__(self, dataset):
        self.sequence_length = dataset['sequence_length']
        self.previous_covariates = dataset['previous_covariates']
        self.previous_treatments = dataset['previous_treatments']
        self.covariates = dataset['covariates']
        self.treatments = dataset['treatments']
        self.outcomes = dataset['outcomes']

    def __getitem__(self, index):
        previous_covariate = self.previous_covariates[index]
        previous_treatment = self.previous_treatments[index]
        covariate = self.covariates[index]
        treatment = self.treatments[index]
        outcome = self.outcomes[index]
        return torch.from_numpy(previous_covariate), torch.from_numpy(previous_treatment), torch.from_numpy(
            covariate), torch.from_numpy(treatment), torch.from_numpy(outcome)

    def __len__(self):
        return self.previous_covariates.shape[0]

def get_dataset_splits(dataset, use_predicted_confounders):
    if use_predicted_confounders:
        dataset_keys = ['previous_covariates', 'previous_treatments', 'covariates', 'treatments',
                        'predicted_confounders', 'outcomes']
    else:
        dataset_keys = ['previous_covariates', 'previous_treatments', 'covariates', 'treatments', 'outcomes']

    num_samples = dataset['previous_covariates'].shape[0]
    train_size = int(num_samples * 0.8)
    val_size = int(num_samples * 0.1)
    test_size = num_samples - train_size - val_size

    train_index = slice(0, train_size)
    val_index = slice(train_size, train_size + val_size)
    test_index = slice(train_size + val_size, num_samples)

    dataset_train = dict()
    dataset_val = dict()
    dataset_test = dict()
    for key in dataset_keys:
        dataset_train[key] = dataset[key][train_index, :, :]
        dataset_val[key] = dataset[key][val_index, :, :]
        dataset_test[key] = dataset[key][test_index, :, :]

    _, length, num_covariates = dataset_train['covariates'].shape

    key = 'sequence_length'
    dataset_train[key] = dataset[key][train_index]
    dataset_val[key] = dataset[key][val_index]
    dataset_test[key] = dataset[key][test_index]

    dataset_map = dict()

    dataset_map['num_time_steps'] = length
    dataset_map['training_data'] = dataset_train
    dataset_map['validation_data'] = dataset_val
    dataset_map['test_data'] = dataset_test

    return dataset_map
'''



import torch
from torch.utils.data import Dataset

class dataset_base(Dataset):
    def __init__(self, dataset):
        self.sequence_length = dataset['sequence_length']
        self.previous_covariates = dataset['previous_covariates']
        self.previous_treatments = dataset['previous_treatments']
        self.covariates = dataset['covariates']
        self.treatments = dataset['treatments']
        self.outcomes = dataset['outcomes']

    def __getitem__(self, index):
        previous_covariate = self.previous_covariates[index]
        previous_treatment = self.previous_treatments[index]
        covariate = self.covariates[index]
        treatment = self.treatments[index]
        outcome = self.outcomes[index]
        return torch.from_numpy(previous_covariate), torch.from_numpy(previous_treatment), torch.from_numpy(covariate), torch.from_numpy(treatment), torch.from_numpy(outcome)

    def __len__(self):
        return self.previous_covariates.shape[0]

def get_dataset_splits(dataset, use_predicted_confounders):
    if use_predicted_confounders:
        dataset_keys = ['previous_covariates', 'previous_treatments', 'covariates', 'treatments', 'predicted_confounders', 'outcomes']
    else:
        dataset_keys = ['previous_covariates', 'previous_treatments', 'covariates', 'treatments', 'outcomes']

    num_time_steps = dataset['previous_covariates'].shape[1]
    
    train_size = int(num_time_steps * 0.8)
    val_size = int(num_time_steps * 0.1)
    test_size = num_time_steps - train_size - val_size

    train_index = slice(0, train_size)
    val_index = slice(train_size, train_size + val_size)
    test_index = slice(train_size + val_size, num_time_steps)

    dataset_train = dict()
    dataset_val = dict()
    dataset_test = dict()
    
    for key in dataset_keys:
        if len(dataset[key].shape) == 3:  # 3D data
            dataset_train[key] = dataset[key][:, train_index, :]
            dataset_val[key] = dataset[key][:, val_index, :]
            dataset_test[key] = dataset[key][:, test_index, :]
        elif len(dataset[key].shape) == 2:  # 2D data
            dataset_train[key] = dataset[key][:, train_index]
            dataset_val[key] = dataset[key][:, val_index]
            dataset_test[key] = dataset[key][:, test_index]
        else:
            dataset_train[key] = dataset[key][train_index]
            dataset_val[key] = dataset[key][val_index]
            dataset_test[key] = dataset[key][test_index]

    length = train_size

    key = 'sequence_length'
    dataset_train[key] = dataset[key][train_index]
    dataset_val[key] = dataset[key][val_index]
    dataset_test[key] = dataset[key][test_index]

    dataset_map = dict()

    dataset_map['num_time_steps'] = length
    dataset_map['training_data'] = dataset_train
    dataset_map['validation_data'] = dataset_val
    dataset_map['test_data'] = dataset_test

    return dataset_map


