import os
import argparse
import logging
import numpy as np
import csv
from scipy.special import expit

from simulated_autoregressive import AutoregressiveSimulation
from time_series_deconfounder import time_series_deconfounder
from utils.evaluation_utils import load_results
from preprocess import process_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", default=0.6, type=float)
    parser.add_argument("--num_simulated_hidden_confounders", default=1, type=int)
    parser.add_argument("--num_substitute_hidden_confounders", default=1, type=int)
    parser.add_argument("--results_dir", default='results')
    parser.add_argument("--exp_name", default='test_tsd_gamma_0.6')
    parser.add_argument("--b_hyperparm_tuning", default=True)
    parser.add_argument("--train_and_get_confounder", action='store_true', default=True)
    parser.add_argument('--num_locations', type=int, default=10, help='Number of locations')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    args = init_arg()

    model_name = 'factor_model'
    hyperparams_file = '{}/{}_best_hyperparams.txt'.format(args.results_dir, model_name)

    logging.info("Creating results directory if it doesn't exist.")
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    #dataset_with_confounders_filename = '{}/{}_dataset_with_substitute_confounders.txt'.format(args.results_dir, args.exp_name)
    dataset_with_confounders_filename = '{}/{}_dataset_with_substitute_confounders.pkl'.format(args.results_dir, args.exp_name)
    factor_model_hyperparams_file = '{}/{}_factor_model_best_hyperparams.txt'.format(args.results_dir, args.exp_name)

    if args.train_and_get_confounder:
        logging.info("Generating synthetic dataset.")
        np.random.seed(2025)
        autoregressive = AutoregressiveSimulation(args.gamma, args.num_simulated_hidden_confounders)

        # Example usage:
        #file_path = '/home/wentao/papercode/iTransformer/dataset/weather/time_merged_observation_ipsl_data.csv'  # Replace with the actual file pathTSD_623/data/South_AU_IPSL_data.csv
        file_path = '/home/wentao/papercode/TSD_623/data/finaldata/merged_data_r1i1p1f1.csv'  # Replace with the actual file path
        dataset = process_dataset(file_path)
        print({k: v.shape for k, v in dataset.items()})
        # Example usage
        # Ensure that dataset['covariates'] is a numpy array before slicing

        logging.info("Climate dataset merged_observation_ipsl processed successfully.")

        logging.info("Starting time series deconfounder process.")
        time_series_deconfounder(dataset=dataset, num_substitute_confounders=args.num_substitute_hidden_confounders,
                                 exp_name=args.exp_name,
                                 dataset_with_confounders_filename=dataset_with_confounders_filename,
                                 factor_model_hyperparams_file=factor_model_hyperparams_file,
                                 b_hyperparm_tuning=args.b_hyperparm_tuning)
        logging.info("Time series deconfounder process comdpleted.")
        

    

    logging.info("Script execution completed.")
