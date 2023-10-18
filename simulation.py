"""
This module contains the main script for the simulation of the stochastic resonance mechanism for the
Earth's climate model.

The script reads a configuration file for the simulation parameters and saves the results in the
specified paths (all this files are contained in the data folder).

"""

import configparser
import sys
import os
import numpy as np
import gc
from scipy import signal

import stochastic_resonance as sr
import aesthetics as aes

#-------Reading parameters values and paths from the configuration file-----------

config = configparser.ConfigParser()

# Reading the configuration file; if it is not specified as an argument, 
# the 'configuration.txt' file is used as default:
config_file = sys.argv[1] if len(sys.argv) > 1 else 'configuration.txt'

if not os.path.isfile(config_file):
    with aes.red_text():
        if config_file == 'configuration.txt':
            print('Error: The default configuration file "configuration.txt" does not exist in the current folder!')
        else:
            print(f'Error: The specified configuration file "{config_file}" does not exist in the current folder!')
        sys.exit()

config.read(config_file)

stable_temperature_solution_1 = config['settings'].getfloat('stable_temperature_solution_1')
unstable_temperature_solution = config['settings'].getfloat('unstable_temperature_solution')
stable_temperature_solution_2 = config['settings'].getfloat('stable_temperature_solution_2')

surface_heat_capacity_j_per_m2_K = config['settings'].getfloat('surface_earth_thermal_capacity')

relaxation_time = config['settings'].getfloat('relaxation_time')
emission_model = config['settings'].get('emission_model')

num_sec_in_a_year = 365.25*24*60*60

C_years = surface_heat_capacity_j_per_m2_K * (num_sec_in_a_year ** 2)

forcing_amplitude = config['settings'].getfloat('forcing_amplitude')
forcing_period = config['settings'].getfloat('forcing_period')

forcing_angular_frequency = (2 * np.pi)/ forcing_period

num_steps = config['settings'].getint('num_steps')
num_simulations = config['settings'].getint('num_simulations')
time_step = config['settings'].getfloat('time_step')

variance_start = config['settings'].getfloat('variance_start')
variance_end = config['settings'].getfloat('variance_end')
num_variances = config['settings'].getint('num_variances')

if 'seed_value' in config['settings']:
    seed_value = config['settings'].getint('seed_value')
    print("The seed value used for this simulation is: ", seed_value)
else: 
    # Generate a random seed value between 0 and 1000000
    seed_value = np.random.randint(0, 1000000)
    with aes.orange_text():
        print('The seed value was not specified in the configuration file.')
    print('The randomly generated seed value used for this simulation is: ', seed_value)

os.makedirs('data', exist_ok = True)

peaks_strengths_in_PSD_destination = config['data_paths'].get('peaks_strengths_in_PSD')
peaks_heights_in_PSD_destination = config['data_paths'].get('peaks_heights_in_PSD')
SNR_destination = config['data_paths'].get('SNR')
results_destination = config['data_paths'].get('results')

def delete_file_if_exists(file_path):
    if os.path.isfile(file_path):
        try:
            os.remove(file_path)
        except:
            with aes.red_text():
                print(f"Error deleting file '{file_path}'.")

delete_file_if_exists(SNR_destination)
delete_file_if_exists(results_destination)

#-----Simulation of temperature evolution for different noise variances---------

print('Simulating temperature evolution for different noise variances...')

V = np.linspace(variance_start, variance_end, num_variances)

peaks_values = np.zeros(len(V))
peaks_height_values = np.zeros(len(V))
SNR_values = np.zeros(len(V))

for i, v in enumerate(V):
    print(f'Simulation {i + 1}/{len(V)}, Noise: {v:.3f}...')
    time, temperature = sr.simulate_ito(T_start = stable_temperature_solution_2,
					noise_variance = v,
					dt = time_step,
					num_steps = num_steps,
					num_simulations = num_simulations,
					surface_thermal_capacity = C_years,
					relaxation_time = relaxation_time,
					stable_temperature_solution_1 = stable_temperature_solution_1,
					unstable_temperature_solution = unstable_temperature_solution,
					stable_temperature_solution_2 = stable_temperature_solution_2,
					forcing_amplitude = forcing_amplitude,
					forcing_angular_frequency = forcing_angular_frequency,
					noise = True,
					emission_model = emission_model,
                    seed_value = seed_value)
    
    print('Binirizing temperature...')

    sr.binarize_temperature_inplace(temperature=temperature,
                                    stable_temperature_solution_1=stable_temperature_solution_1,
                                    stable_temperature_solution_2=stable_temperature_solution_2)

    print('Calculating PSD...')

    psd = np.zeros((num_simulations, np.floor_divide(num_steps, 2) + 1))
    for j in range(num_simulations):
        frequencies, power_spectrum = signal.periodogram(temperature[j, :], 1/time_step)
        psd[j, :] = power_spectrum
    PSD_mean = np.mean(psd, axis = 0)
    Frequencies = frequencies
    
    peak_index = sr.find_peak_indices(Frequencies, forcing_period)
    peak = sr.calculate_peaks(PSD_mean, peak_index)
    peak_base = sr.calculate_peaks_base(PSD_mean, peak_index)
    peak_height = sr.calculate_peak_height(peak, peak_base)
    SNR = sr.calculate_SNR(peak, peak_base)

    peaks_values[i] = peak
    peaks_height_values[i] = peak_height
    SNR_values[i] = SNR
    del time, temperature, psd, PSD_mean, Frequencies, frequencies, power_spectrum, \
        peak_index, peak, peak_base, peak_height, SNR
    gc.collect()

np.save(SNR_destination, SNR_values)

with aes.green_text():
    print('SNR values saved!')

V_sr_index = np.argmax(peaks_height_values)
V_sr = V[V_sr_index]

V_SR_index = np.argmax(SNR_values)
V_SR = V[V_SR_index]

results = np.array([V_sr, V_SR])

print(f'The value of the noise variance that maximizes the peak height is : {V_sr:.3f}')
print(f'The value of the noise variance that maximizes the Signal-to-Noise Ratio is : {V_SR:.3f}')

np.save(results_destination, results)
np.save(peaks_strengths_in_PSD_destination, peaks_values)
np.save(peaks_heights_in_PSD_destination, peaks_height_values)

with aes.green_text():
    print('Results saved!')