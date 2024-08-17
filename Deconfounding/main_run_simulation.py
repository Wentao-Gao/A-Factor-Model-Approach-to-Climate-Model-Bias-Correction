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
        #dataset = autoregressive.generate_dataset(50, 365)
        num_patients = 50 # 你可以根据需要调整患者数量
        max_timesteps = 365 # 你可以根据需要调整最大时间步数
        dataset = autoregressive.generate_dataset(num_patients, max_timesteps)

        ###################################################################################################
        ###################################################################################################
        ###################################################################################################

        # 定义要保存的数据的文件名
        output_file_combined = 'combined_generated_dataset.csv'
        output_file_source1 = 'source1_generated_dataset.csv'
        output_file_source2 = 'source2_generated_dataset.csv'

        # 保存 combined 数据
        with open(output_file_combined, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # 写入表头
            header = ['patient_id', 'time_step']
            num_covariates = dataset['covariates'].shape[2] // 2
            num_confounders = dataset['confounders'].shape[2]
            num_treatments = dataset['treatments'].shape[2] // 2
            
            for i in range(num_covariates):
                header.append(f'covariate_source1_{i+1}')
            for i in range(num_covariates):
                header.append(f'covariate_source2_{i+1}')
            for i in range(num_confounders):
                header.append(f'confounder{i+1}')
            for i in range(num_treatments):
                header.append(f'treatment_source1_{i+1}')
            for i in range(num_treatments):
                header.append(f'treatment_source2_{i+1}')
            header.append('outcome_source1')
            header.append('outcome_source2')
            
            writer.writerow(header)
            
            # 写入数据
            num_patients = dataset['covariates'].shape[0]
            for patient in range(num_patients):
                for timestep in range(max_timesteps):
                    row = [patient, timestep]
                    if timestep < dataset['covariates'][patient].shape[0]:
                        row.extend(dataset['covariates'][patient][timestep][:num_covariates].tolist())
                        row.extend(dataset['covariates'][patient][timestep][num_covariates:].tolist())
                    else:
                        row.extend([0] * (num_covariates * 2))
                    if timestep < dataset['confounders'][patient].shape[0]:
                        row.extend(dataset['confounders'][patient][timestep].tolist())
                    else:
                        row.extend([0] * num_confounders)
                    if timestep < dataset['treatments'][patient].shape[0]:
                        row.extend(dataset['treatments'][patient][timestep][:num_treatments].tolist())
                        row.extend(dataset['treatments'][patient][timestep][num_treatments:].tolist())
                    else:
                        row.extend([0] * (num_treatments * 2))
                    if timestep < dataset['outcomes'][patient].shape[0]:
                        row.append(dataset['outcomes'][patient][timestep][0])
                        row.append(dataset['outcomes'][patient][timestep][1])
                    else:
                        row.extend([0, 0])
                    writer.writerow(row)

        # 保存 source1 数据
        with open(output_file_source1, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # 写入表头
            header = ['patient_id', 'time_step']
            for i in range(num_covariates):
                header.append(f'covariate_source1_{i+1}')
            for i in range(num_confounders):
                header.append(f'confounder{i+1}')
            for i in range(num_treatments):
                header.append(f'treatment_source1_{i+1}')
            header.append('outcome_source1')
            
            writer.writerow(header)
            
            # 写入数据
            num_patients = dataset['covariates'].shape[0]
            for patient in range(num_patients):
                for timestep in range(max_timesteps):
                    row = [patient, timestep]
                    if timestep < dataset['covariates'][patient].shape[0]:
                        row.extend(dataset['covariates'][patient][timestep][:num_covariates].tolist())
                    else:
                        row.extend([0] * num_covariates)
                    if timestep < dataset['confounders'][patient].shape[0]:
                        row.extend(dataset['confounders'][patient][timestep].tolist())
                    else:
                        row.extend([0] * num_confounders)
                    if timestep < dataset['treatments'][patient].shape[0]:
                        row.extend(dataset['treatments'][patient][timestep][:num_treatments].tolist())
                    else:
                        row.extend([0] * num_treatments)
                    if timestep < dataset['outcomes'][patient].shape[0]:
                        row.append(dataset['outcomes'][patient][timestep][0])
                    else:
                        row.append(0)
                    writer.writerow(row)

        # 保存 source2 数据
        with open(output_file_source2, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # 写入表头
            header = ['patient_id', 'time_step']
            for i in range(num_covariates):
                header.append(f'covariate_source2_{i+1}')
            for i in range(num_confounders):
                header.append(f'confounder{i+1}')
            for i in range(num_treatments):
                header.append(f'treatment_source2_{i+1}')
            header.append('outcome_source2')
            
            writer.writerow(header)
            
            # 写入数据
            num_patients = dataset['covariates'].shape[0]
            for patient in range(num_patients):
                for timestep in range(max_timesteps):
                    row = [patient, timestep]
                    if timestep < dataset['covariates'][patient].shape[0]:
                        row.extend(dataset['covariates'][patient][timestep][num_covariates:].tolist())
                    else:
                        row.extend([0] * num_covariates)
                    if timestep < dataset['confounders'][patient].shape[0]:
                        row.extend(dataset['confounders'][patient][timestep].tolist())
                    else:
                        row.extend([0] * num_confounders)
                    if timestep < dataset['treatments'][patient].shape[0]:
                        row.extend(dataset['treatments'][patient][timestep][num_treatments:].tolist())
                    else:
                        row.extend([0] * num_treatments)
                    if timestep < dataset['outcomes'][patient].shape[0]:
                        row.append(dataset['outcomes'][patient][timestep][1])
                    else:
                        row.append(0)
                    writer.writerow(row)

        print(f"Combined 数据已保存到 {output_file_combined}")
        print(f"Source1 数据已保存到 {output_file_source1}")
        print(f"Source2 数据已保存到 {output_file_source2}")
        ###################################################################################################
        ###################################################################################################
        ###################################################################################################


        logging.info("Climate dataset merged_observation_ipsl processed successfully.")

        logging.info("Starting time series deconfounder process.")
        time_series_deconfounder(dataset=dataset, num_substitute_confounders=args.num_substitute_hidden_confounders,
                                 exp_name=args.exp_name,
                                 dataset_with_confounders_filename=dataset_with_confounders_filename,
                                 factor_model_hyperparams_file=factor_model_hyperparams_file,
                                 b_hyperparm_tuning=args.b_hyperparm_tuning)
        logging.info("Time series deconfounder process comdpleted.")
        

    
