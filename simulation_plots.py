import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import ScalarFormatter
import configparser
import sys

import aesthetics as aes

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

variance_start = config['settings'].getfloat('variance_start')
variance_end = config['settings'].getfloat('variance_end')
num_variances = config['settings'].getint('num_variances')
V = np.linspace(variance_start, variance_end, num_variances)

def plot(variance, mean, standard_deviation, title, ylabel):
    
    f = plt.figure(figsize=(15, 10))
    plt.scatter(variance, mean, label= 'Mean values')
    plt.errorbar(variance, mean, yerr=standard_deviation, fmt='o', label='Error bars')

    plt.xlabel(r'Noise variance $\left[ \frac{K^2}{year} \right]$')
    plt.ylabel(ylabel)
    plt.title(title, fontsize=15, fontweight='bold')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    plt.savefig('./plots/' + title + '.png')

# Data directory
directory = './simulation_data' # Directory containing the data files

# Create a list of the files in the directory
for filename in os.listdir(directory):
    if filename.startswith('SNR_') and filename.endswith('.npy'):
        # Extract the number from the filename
        seed_value = int(filename.split('_')[1].split('.')[0])
        
        # Load the data from the file
        data_SNR = np.load(os.path.join(directory, filename))
        
        # Create a variable name from the number and assign the data to it
        variable_name = f'SNR_{seed_value}'
        globals()[variable_name] = data_SNR

SNR_variables = [var for var in globals() if var.startswith('SNR_')]

SNR = np.zeros((len(SNR_variables), num_variances))

for i, variable in enumerate(SNR_variables):
    SNR[i, :] = globals()[variable]

# Calculate the mean and standard deviation of the SNR values
SNR_mean = np.mean(SNR, axis = 0)
SNR_std = np.std(SNR, axis = 0)

for filename in os.listdir(directory):
    if filename.startswith('peaks_strengths_in_PSD_') and filename.endswith('.npy'):
        # Extract the number from the filename
        seed_value = int(filename.split('_')[4].split('.')[0])
        
        # Load the data from the file
        data_peaks_strengths_in_PSD = np.load(os.path.join(directory, filename))
        
        # Create a variable name from the number and assign the data to it
        variable_name = f'peaks_strengths_in_PSD_{seed_value}'
        globals()[variable_name] = data_peaks_strengths_in_PSD

peaks_strengths_in_PSD_variables = [var for var in globals() if var.startswith('peaks_strengths_in_PSD_')]
peaks_strengths_in_PSD = np.zeros((len(peaks_strengths_in_PSD_variables), num_variances))

for i, variable in enumerate(peaks_strengths_in_PSD_variables):
    peaks_strengths_in_PSD[i, :] = globals()[variable]

# Calculate the mean and standard deviation of the SNR values
peaks_strengths_in_PSD_mean = np.mean(peaks_strengths_in_PSD, axis = 0)
peaks_strengths_in_PSD_std = np.std(peaks_strengths_in_PSD, axis = 0)

for filename in os.listdir(directory):
    if filename.startswith('peaks_heights_in_PSD_') and filename.endswith('.npy'):
        # Extract the number from the filename
        seed_value = int(filename.split('_')[4].split('.')[0])
        
        # Load the data from the file
        data_peaks_heights_in_PSD = np.load(os.path.join(directory, filename))
        
        # Create a variable name from the number and assign the data to it
        variable_name = f'peaks_heights_in_PSD_{seed_value}'
        globals()[variable_name] = data_peaks_heights_in_PSD

peaks_heights_in_PSD_variables = [var for var in globals() if var.startswith('peaks_heights_in_PSD_')]
peaks_heights_in_PSD = np.zeros((len(peaks_heights_in_PSD_variables), num_variances))

for i, variable in enumerate(peaks_heights_in_PSD_variables):
    peaks_heights_in_PSD[i, :] = globals()[variable]

# Calculate the mean and standard deviation of the SNR values
peaks_heights_in_PSD_mean = np.mean(peaks_heights_in_PSD, axis = 0)
peaks_heights_in_PSD_std = np.std(peaks_heights_in_PSD, axis = 0)

# Create the plots directory if it does not exist
os.makedirs('plots', exist_ok = True)

# Plot the SNR
plot(
    variance = V, 
    mean = SNR_mean, 
    standard_deviation = SNR_std, 
    title = 'Signal-to-noise ratio as a function of the noise variance', 
    ylabel = 'SNR'
)
# Plot the peaks strengths in the PSD
plot(
    variance = V, 
    mean = peaks_strengths_in_PSD_mean, 
    standard_deviation = peaks_strengths_in_PSD_std, 
    title = 'Peaks strengths in the PSD as a function of the noise variance', 
    ylabel = 'Peaks strengths in the PSD'
)
# Plot the peaks heights in the PSD
plot(
    variance = V, 
    mean = peaks_heights_in_PSD_mean, 
    standard_deviation = peaks_heights_in_PSD_std, 
    title = 'Peaks heights in the PSD as a function of the noise variance', 
    ylabel = 'Peaks heights in the PSD'
)

with aes.green_text():
    print('Plots saved!')

