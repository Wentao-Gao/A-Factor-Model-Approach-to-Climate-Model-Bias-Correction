import csv
import numpy as np
from scipy.special import expit

class AutoregressiveSimulation:
    def __init__(self, gamma, num_simulated_hidden_confounders, seed=None):
        if seed is not None:
            np.random.seed(seed)  # Set up random seed
        self.num_covariates = 3
        self.num_confounders = num_simulated_hidden_confounders
        self.num_treatments = 3
        self.p = 5

        self.gamma_a = gamma
        self.gamma_y = gamma

        self.covariates_coefficients = dict()
        self.covariates_coefficients['source1_treatments'] = self.generate_coefficients(
            self.p, matrix_shape=(self.num_covariates, self.num_treatments), treatment_coefficients=True)
        self.covariates_coefficients['source2_treatments'] = self.generate_coefficients(
            self.p, matrix_shape=(self.num_covariates, self.num_treatments), treatment_coefficients=True)

        self.confounders_coefficients = dict()
        self.confounders_coefficients['source1_treatments'] = self.generate_coefficients(
            self.p, matrix_shape=(self.num_confounders, self.num_treatments))
        self.confounders_coefficients['source2_treatments'] = self.generate_coefficients(
            self.p, matrix_shape=(self.num_confounders, self.num_treatments))
        self.confounders_coefficients['confounders'] = self.generate_coefficients(
            self.p, matrix_shape=(self.num_confounders, self.num_confounders), variables_coefficients=True)

        self.outcome_coefficients = np.array([np.random.normal(0, 1) for _ in range(self.num_confounders + self.num_covariates)])
        self.treatment_coefficients = self.generate_treatment_coefficients()

    def generate_treatment_coefficients(self):
        treatment_coefficients = np.zeros(shape=(self.num_treatments, self.num_covariates + self.num_confounders))
        for treatment in range(self.num_treatments):
            treatment_coefficients[treatment][treatment] = 1.0 - self.gamma_a
            treatment_coefficients[treatment][self.num_covariates] = self.gamma_a
        return treatment_coefficients

    def generate_coefficients(self, p, matrix_shape, variables_coefficients=False, treatment_coefficients=False):
        coefficients = []
        for i in range(p):
            if variables_coefficients:
                diag_elements = [np.random.normal(1.0 - (i+1) * 0.2, 0.2) for _ in range(matrix_shape[0])]
                timestep_coefficients = np.diag(diag_elements)
            elif treatment_coefficients:
                diag_elements = [np.random.normal(0, 0.5) for _ in range(matrix_shape[1])]
                timestep_coefficients = np.diag(diag_elements)
            else:
                timestep_coefficients = np.random.normal(0, 0.5, size=matrix_shape[1])
            normalized_coefficients = timestep_coefficients / p
            coefficients.append(normalized_coefficients)
        return coefficients

    def generate_treatment_assignments_single_timestep(self, p, history, source):
        confounders_history = history['confounders']
        covariates_history = history[source]['covariates']
        history_length = len(covariates_history)
        if history_length < p:
            p = history_length
        average_covariates = np.zeros(shape=len(covariates_history[-1]))
        average_confounders = np.zeros(shape=len(confounders_history[-1]))
        for index in range(p):
            average_covariates += covariates_history[history_length - index - 1]
            average_confounders += confounders_history[history_length - index - 1]
        all_variables = np.concatenate((average_covariates, average_confounders)).T

        treatment_assignment = np.zeros(shape=(self.num_treatments,))
        lambda_value = 2 if source == 'source1' else 2 #0.5
        for index in range(self.num_treatments):
            aux_normal = lambda_value * np.dot(all_variables, self.treatment_coefficients[index])
            treatment_assignment[index] = aux_normal  # Using continuous value instead of binary.

        return treatment_assignment
    

    def generate_covariates_single_timestep(self, history, source):
        treatments_history = history[source]['treatments']
        history_length = len(treatments_history)
        if history_length > 0:
            x_t = treatments_history[-1]
        else:
            x_t = np.zeros(shape=(self.num_covariates,))
        # Generate different noise for differen source
        noise = np.random.normal(0 if source == 'source1' else 1, 0.001 if source == 'source1' else 0.002, size=(self.num_covariates))
        noise = np.random.normal(0, 0.001 if source == 'source1' else 0.002, size=(self.num_covariates))
        x_t += noise
        x_t = np.clip(x_t, -1, 1)
        return x_t

    def generate_confounders_single_timestep(self, p, history):
        treatments_history_source1 = history['source1']['treatments']
        treatments_history_source2 = history['source2']['treatments']
        confounders_history = history['confounders']
        past_treatment_coefficients_source1 = self.confounders_coefficients['source1_treatments']
        past_treatment_coefficients_source2 = self.confounders_coefficients['source2_treatments']
        past_confounders_coefficients = self.confounders_coefficients['confounders']
        history_length = len(confounders_history)
        if history_length < p:
            p = history_length
        treatments_sum = np.zeros(shape=(self.num_confounders,))
        confounders_sum = np.zeros(shape=(self.num_confounders,))
        for index in range(p):
            treatments_sum += np.matmul(past_treatment_coefficients_source1[index], treatments_history_source1[history_length - index - 1])
            treatments_sum += np.matmul(past_treatment_coefficients_source2[index], treatments_history_source2[history_length - index - 1])
            confounders_sum += np.matmul(past_confounders_coefficients[index], confounders_history[history_length - index - 1])
        noise = np.random.normal(0, 0.001, size=(self.num_confounders))
        z_t = treatments_sum + confounders_sum + noise
        z_t = np.clip(z_t, -1, 1)
        return z_t

    def generate_data_single_location(self, timesteps):
        x_0_source1 = np.random.normal(0, 1, size=(self.num_covariates,))
        x_0_source2 = np.random.normal(0, 1, size=(self.num_covariates,))
        z_0 = np.random.normal(0, 1, size=(self.num_confounders,))
        a_0_source1 = np.zeros(shape=(self.num_treatments,))
        a_0_source2 = np.zeros(shape=(self.num_treatments,))
        history = dict()
        history['source1'] = {'covariates': [x_0_source1], 'treatments': [a_0_source1]}
        history['source2'] = {'covariates': [x_0_source2], 'treatments': [a_0_source2]}
        history['confounders'] = [z_0]
        for t in range(timesteps):
            x_t_source1 = self.generate_covariates_single_timestep(history, 'source1')
            x_t_source2 = self.generate_covariates_single_timestep(history, 'source2')
            z_t = self.generate_confounders_single_timestep(self.p, history)
            history['source1']['covariates'].append(x_t_source1)
            history['source2']['covariates'].append(x_t_source2)
            history['confounders'].append(z_t)
            a_t_source1 = self.generate_treatment_assignments_single_timestep(self.p, history, 'source1')
            a_t_source2 = self.generate_treatment_assignments_single_timestep(self.p, history, 'source2')
            history['source1']['treatments'].append(a_t_source1)
            history['source2']['treatments'].append(a_t_source2)
        return history

    def generate_dataset(self, num_patients, max_timesteps):
        dataset = {'previous_covariates': [], 'previous_treatments': [], 'covariates': [], 'confounders': [], 'treatments': [], 'sequence_length': [], 'outcomes': []}
        for patient in range(num_patients):
            timesteps = np.random.randint(int(max_timesteps)-10, int(max_timesteps), 1)[0]
            history = self.generate_data_single_location(timesteps + 1)
            previous_covariates = np.hstack((np.vstack((np.array(history['source1']['covariates'][1:timesteps]), np.zeros(shape=(max_timesteps-timesteps, self.num_covariates)))), 
                                             np.vstack((np.array(history['source2']['covariates'][1:timesteps]), np.zeros(shape=(max_timesteps-timesteps, self.num_covariates))))))
            previous_treatments = np.hstack((np.vstack((np.array(history['source1']['treatments'][1:timesteps]), np.zeros(shape=(max_timesteps-timesteps, self.num_treatments)))), 
                                             np.vstack((np.array(history['source2']['treatments'][1:timesteps]), np.zeros(shape=(max_timesteps-timesteps, self.num_treatments))))))
            covariates = np.hstack((np.vstack((np.array(history['source1']['covariates'][1:timesteps+1]), np.zeros(shape=(max_timesteps - timesteps, self.num_covariates)))), 
                                    np.vstack((np.array(history['source2']['covariates'][1:timesteps+1]), np.zeros(shape=(max_timesteps - timesteps, self.num_covariates))))))
            treatments = np.hstack((np.vstack((np.array(history['source1']['treatments'][1:timesteps+1]), np.zeros(shape=(max_timesteps-timesteps, self.num_treatments)))), 
                                    np.vstack((np.array(history['source2']['treatments'][1:timesteps+1]), np.zeros(shape=(max_timesteps-timesteps, self.num_treatments))))))
            confounders = np.vstack((np.array(history['confounders'][1:timesteps+1]), np.zeros(shape=(max_timesteps - timesteps, self.num_confounders))))
            outcomes = np.hstack((self.gamma_y * np.mean(np.array(history['confounders'][1:timesteps+1]), axis=-1)[:, np.newaxis] + (1-self.gamma_y) * np.mean(np.array(history['source1']['covariates'][1:timesteps+1]), axis=-1)[:, np.newaxis],
                                  self.gamma_y * np.mean(np.array(history['confounders'][1:timesteps+1]), axis=-1)[:, np.newaxis] + (1-self.gamma_y) * np.mean(np.array(history['source2']['covariates'][1:timesteps+1]), axis=-1)[:, np.newaxis]))
            outcomes = np.vstack((outcomes, np.zeros(shape=(max_timesteps-timesteps, 2))))
            dataset['previous_covariates'].append(previous_covariates)
            dataset['previous_treatments'].append(previous_treatments)
            dataset['covariates'].append(covariates)
            dataset['confounders'].append(confounders)
            dataset['treatments'].append(treatments)
            dataset['sequence_length'].append(timesteps)
            dataset['outcomes'].append(outcomes)
        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])
        return dataset
 