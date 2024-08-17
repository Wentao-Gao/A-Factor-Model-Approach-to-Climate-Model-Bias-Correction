import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def process_dataset(file_path, missing_threshold=0.5, zero_threshold=0.5):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Drop features with high missing or zero values
    #data = data.drop(columns=['sfcr_sfc', 'nswrs_sfc', 'nlwrs_sfc', 'pevpr_sfc', 'gflux_sfc', 'runof_sfc'])

    # Drop rows with any missing values
    #data = data.dropna()

    # Detect features with high missing or zero values
    missing_ratio = data.isnull().mean()
    zero_ratio = (data == 0).mean()

    # Identify columns to drop based on the thresholds
    cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index.union(
                   zero_ratio[zero_ratio > zero_threshold].index)

    
    # Convert time column to datetime
    data['time'] = pd.to_datetime(data['time'])

    # Initialize the dataset dictionary
    dataset_corrected = {
        'previous_covariates': [],
        'previous_treatments': [],
        'covariates': [],
        'confounders': [],
        'treatments': [],
        'sequence_length': [],
        'outcomes': [],
        'time': [],  # Add a key to store time information
        'lat': [],   # Add a key to store latitude information
        'lon': []    # Add a key to store longitude information
    }

    # Group by lat and lon
    locations = data.groupby(['lat', 'lon'])

    # Find the maximum number of time steps across all locations
    max_timesteps = 0
    for _, group in locations:
        max_timesteps = max(max_timesteps, len(group))

    # Iterate over each unique location
    for (lat, lon), group in locations:
        group = group.sort_values('time')

        # Extract covariates (treatments)
        covariates = group.drop(columns=['lat', 'lon', 'time', 'prate']).values
        outcomes = group['prate'].values
        time_values = group['time'].values  # Extract time values

        # Initialize history lists
        previous_covariates = []
        previous_treatments = []
        covariates_list = []
        treatments_list = []
        outcomes_list = []
        confounders_list = []
        time_list = []  # List to store time information

        num_timesteps = len(group)

        # Standardize the covariates (treatments)
        scaler = StandardScaler()
        covariates = scaler.fit_transform(covariates)

        # Iterate through the time steps
        for t in range(2, num_timesteps):
            previous_covariates.append(covariates[t - 2])  # t-2 step covariates
            previous_treatments.append(covariates[t - 1])  # t-1 step covariates
            covariates_list.append(covariates[t - 1])      # t-1 step covariates
            treatments_list.append(covariates[t])          # current covariates
            outcomes_list.append([outcomes[t]])            # current outcomes
            confounders_list.append([0])                   # Placeholder for confounders
            time_list.append(time_values[t])               # Store current time

        # Convert lists to numpy arrays
        previous_covariates = np.array(previous_covariates)
        previous_treatments = np.array(previous_treatments)
        covariates_list = np.array(covariates_list)
        treatments_list = np.array(treatments_list)
        outcomes_list = np.array(outcomes_list)
        confounders_list = np.array(confounders_list)
        time_list = np.array(time_list)  # Convert time list to numpy array

        # Pad sequences to the maximum time steps
        pad_length_prev = max_timesteps - 2  # For previous_* arrays
        pad_length = max_timesteps - 1       # For the rest of the arrays
        previous_covariates = np.pad(previous_covariates, ((0, pad_length_prev - len(previous_covariates)), (0, 0)), 'constant')
        previous_treatments = np.pad(previous_treatments, ((0, pad_length_prev - len(previous_treatments)), (0, 0)), 'constant')
        covariates_list = np.pad(covariates_list, ((0, pad_length - len(covariates_list)), (0, 0)), 'constant')
        treatments_list = np.pad(treatments_list, ((0, pad_length - len(treatments_list)), (0, 0)), 'constant')
        outcomes_list = np.pad(outcomes_list, ((0, pad_length - len(outcomes_list)), (0, 0)), 'constant')
        confounders_list = np.pad(confounders_list, ((0, pad_length - len(confounders_list)), (0, 0)), 'constant')
        time_list = np.pad(time_list, (0, pad_length - len(time_list)), 'constant', constant_values=np.datetime64('NaT'))  # Pad time with NaT

        dataset_corrected['previous_covariates'].append(previous_covariates)
        dataset_corrected['previous_treatments'].append(previous_treatments)
        dataset_corrected['covariates'].append(covariates_list)
        dataset_corrected['treatments'].append(treatments_list)
        dataset_corrected['confounders'].append(confounders_list)
        dataset_corrected['outcomes'].append(outcomes_list)
        dataset_corrected['sequence_length'].append(num_timesteps - 2)
        dataset_corrected['time'].append(time_list)  # Add time information
        dataset_corrected['lat'].append([lat] * len(time_list))  # Add latitude information
        dataset_corrected['lon'].append([lon] * len(time_list))  # Add longitude information

    # Convert lists to numpy arrays
    for key in dataset_corrected:
        try:
            dataset_corrected[key] = np.array(dataset_corrected[key])
        except ValueError as e:
            print(f"Error converting {key} to numpy array: {e}")
            if len(dataset_corrected[key]) > 1:
                print(f"First element shape: {np.array(dataset_corrected[key][0]).shape}")
                print(f"Second element shape: {np.array(dataset_corrected[key][1]).shape}")

    return dataset_corrected
