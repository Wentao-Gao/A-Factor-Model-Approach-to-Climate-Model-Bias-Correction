import torch.nn
import torch
import numpy as np
import os
import shutil
from sklearn.model_selection import ShuffleSplit
import logging
from utils.torch_utils import dataset_base, get_dataset_splits
from factor_model import FactorModel
from torch.utils.data import DataLoader
from utils.evaluation_utils import write_results_to_file
import matplotlib.pyplot as plt
import pickle

def train_factor_model(dataset_train, dataset_val, dataset, num_confounders, hyperparams_file,
                       b_hyperparameter_optimisation):
    _, length, num_covariates = dataset_train['covariates'].shape
    num_treatments = dataset_train['treatments'].shape[-1]
    device = 'cuda'
    params = {'num_treatments': num_treatments,
              'num_covariates': num_covariates,
              'num_confounders': num_confounders,
              'max_sequence_length': length,
              'num_epochs': 50}

    best_hyperparams = {
        'rnn_hidden_units': 128,
        'fc_hidden_units': 128, #64 for real data
        'learning_rate': 0.001,
        'batch_size':16,#2048
        'rnn_keep_prob': 0.8}

    trainset = dataset_base(dataset_train)
    valset = dataset_base(dataset_val)
    allset = dataset_base(dataset)
    trainloader = DataLoader(trainset, batch_size=best_hyperparams['batch_size'], shuffle=True)
    valloader = DataLoader(valset, batch_size=best_hyperparams['batch_size'], shuffle=False)
    allloader = DataLoader(allset, batch_size=best_hyperparams['batch_size'], shuffle=False)

    factor_model = FactorModel(params=params, hyperparams=best_hyperparams, device=device)
    factor_model.train_model(trainloader, valloader)
    #r2_scores_over_time = factor_model.train_model(trainloader, valloader)
    predicted_confounders = factor_model.compute_hidden_confounders(allloader)



    return predicted_confounders

def time_series_deconfounder(dataset, num_substitute_confounders, exp_name, dataset_with_confounders_filename,
                             factor_model_hyperparams_file, b_hyperparm_tuning=False):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    dataset_map = get_dataset_splits(dataset, use_predicted_confounders=False)
    
    dataset_train = dataset_map['training_data']
    dataset_val = dataset_map['validation_data']

    logging.info("Fitting factor model")
    predicted_confounders = train_factor_model(dataset_train, dataset_val,
                                               dataset,
                                               num_confounders=1, #num_substitute_confounders
                                               b_hyperparameter_optimisation=b_hyperparm_tuning,
                                               hyperparams_file=factor_model_hyperparams_file)
    dataset['predicted_confounders'] = predicted_confounders
    # write_results_to_file(dataset_with_confounders_filename, dataset)
    # using pickle to save 
    with open(dataset_with_confounders_filename, 'wb') as f:
        pickle.dump(dataset, f)
    logging.info(f"Dataset with confounders saved to {dataset_with_confounders_filename}")
