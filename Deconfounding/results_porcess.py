import pandas as pd
import numpy as np
import pickle

file_path = 'test_tsd_gamma_0.6_dataset_with_substitute_confounders.pkl'
with open(file_path, 'rb') as file:
    data = pickle.load(file)

predicted_confounders = data['predicted_confounders']

predicted_confounders_shape = predicted_confounders.shape
print(predicted_confounders_shape)

data_list = []

for patient_id in range(predicted_confounders.shape[0]):
    for time_step in range(predicted_confounders.shape[1]):
        predicted_confounder_value = predicted_confounders[patient_id, time_step, 0]
        data_list.append([patient_id, time_step, predicted_confounder_value])

df = pd.DataFrame(data_list, columns=['patient_id', 'time_step', 'predicted_confounder'])

output_file_path = 'predicted_confounders.csv'
df.to_csv(output_file_path, index=False)

output_file_path
