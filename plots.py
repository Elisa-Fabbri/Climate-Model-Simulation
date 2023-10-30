"""
This module contains the functions for plotting the results of the simulation.

It reads the data from the data folder and generates plots in the images folder.
"""
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

try:
    peak_strengths_in_PSD_path = config['data_paths'].get('peaks_strengths_in_PSD')
    peak_heights_in_PSD_path = config['data_paths'].get('peaks_heights_in_PSD')
    pseudo_potential_plot_path = config['data_paths'].get('pseudo_potential')

    SNR_path = config['data_paths'].get('SNR')

except:
    with aes.red_text():
        print("An error occurred while reading data paths from the configuration file.")
    with aes.orange_text():
        print("Please make sure you have correctly specified the data paths in the configuration file.")
    sys.exit(1)


os.makedirs('images', exist_ok = True)
os.makedirs('images/additional_images', exist_ok = True)

try:
    peak_strengths_plot_destination = config['image_paths'].get('peak_strengths_plot')
    peak_heights_plot_destination = config['image_paths'].get('peak_heights_plot')
    SNR_plot_destination = config['image_paths'].get('SNR_plot')
    pseudo_potential_plot_destination = config['image_paths'].get('pseudo_potential_plot')
except:
    with aes.red_text():
        print("An error occurred while reading image paths from the configuration file.")
    with aes.orange_text():
        print("Please make sure you have correctly specified the image paths in the configuration file.")


def check_path_and_load_data(data_path, contained_data):
    """
    Check if the specified data path exists and load the data.

    Parameters:
    - data_path: The path to the data file.
    - contained_data: The type of data contained in the file.

    Returns:
    - data: The data stored in the file.

    Raises:
    - FileNotFoundError: If the file does not exist.
    - TypeError: If the data path is not specified correctly in the configuration file.
    """
    try:
        data = np.load(data_path)
    except FileNotFoundError:
        with aes.red_text():
            print(f"An error occurred: The file containing the '{contained_data}' data does not exist.")
        print("To generate this file, please run 'simulation.py' first.")
        sys.exit(1)
    except TypeError:
        with aes.red_text():
            print("An error occurred while reading data paths from the configuration file.")
        with aes.orange_text():
            print("Please make sure you have correctly specified the data paths in the configuration file.")
        sys.exit()
    return data


def power_spectra_plots(frequencies_path = 'data/frequencies.npy',
                        averaged_PSD_path = 'data/averaged_PSD.npy',
                        power_spectra_plots_destination = 'images/power_spectra.png'):
    """
    Plot power spectral density as a function of frequency for different noise variance values.

    This function generates a set of plots, each showing the computed power spectral density (PSD) 
    as a function of frequency for different values of the noise variance.

    """

    frequencies = check_path_and_load_data(data_path = frequencies_path, 
                                           contained_data = 'frequency')
    
    PSD_mean = check_path_and_load_data(data_path = averaged_PSD_path, 
                                        contained_data = 'power spectral density')

    variance_start = config['settings'].getfloat('variance_start')
    variance_end = config['settings'].getfloat('variance_end')
    num_variances = config['settings'].getint('num_variances')

    V = np.linspace(variance_start, variance_end, num=num_variances)

    if frequencies.shape[0] != PSD_mean.shape[0]:
        raise ValueError("The dimensions of the two arrays do not align.")

    num_plots = frequencies.shape[0]
    num_cols = num_plots // 2
    num_rows = (num_plots // num_cols) + int(num_plots % num_cols != 0)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(30, 10))

    for i in range(num_plots):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]

        ax.semilogy(frequencies[i], PSD_mean[i])
        ax.set_xlabel(r'Frequency $\left[ \frac{1}{year} \right]$', fontsize=11)
        ax.set_ylabel('PSD', fontsize=11)
        ax.set_title('Variance: {0}'.format(round(V[i], 3)), fontsize=13, fontweight='bold')
        ax.set_xlim(0, 2.5e-5)
        ax.set_ylim(1e-1, 1e8)
        ax.grid(True)
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=True))

    title = 'Power Spectral Density as a function of the Frequency for different Noise Variance Values'
    plt.suptitle(title, fontsize=20, fontweight='bold')

    caption = 'The plots show the computed power spectral density for different values of the noise variance'
    fig.text(0.5, 0.01, caption, horizontalalignment='center', fontsize=12, linespacing=0.8, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(power_spectra_plots_destination)

def peak_strenght_plot():
    """
    Plot the strength of the peak as a function of the noise variance.

    This function generates a scatter plot illustrating the strength of the peak in the power spectrum
    computed for various noise variance values. The peak strength is determined by measuring the peak
    values in the power sprectrum.
    """

    variance_start = config['settings'].getfloat('variance_start')
    variance_end = config['settings'].getfloat('variance_end')
    num_variances = config['settings'].getint('num_variances')

    V = np.linspace(variance_start, variance_end, num=num_variances)

    peaks_strength = check_path_and_load_data(data_path = peak_strengths_in_PSD_path,
                                            contained_data = 'peak strengths')
    
    f = plt.figure(figsize=(15, 10))
    plt.scatter(V, peaks_strength)
    plt.xlabel(r'Noise variance $\left[ \frac{K^2}{year} \right]$')
    plt.ylabel(r'Peak strength $\left[ K^2 \cdot year \right]$')
    plt.title('Peak strength as a function of the noise variance', fontsize=15, fontweight='bold')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=True))
    plt.savefig(peak_strengths_plot_destination)

def peak_height_plot():
    """
    Plot the height of the peak as a function of the noise variance.

    This function generates a scatter plot illustrating the height of the peak in the power spectrum 
    computed for various noise variance values. The peak height is determined by measuring the peak 
    height in the power spectrum and subtracting the baseline height.

    """

    variance_start = config['settings'].getfloat('variance_start')
    variance_end = config['settings'].getfloat('variance_end')
    num_variances = config['settings'].getint('num_variances')

    V = np.linspace(variance_start, variance_end, num=num_variances)

    peaks_height = check_path_and_load_data(data_path = peak_heights_in_PSD_path,
                                            contained_data = 'peak heights')

    f = plt.figure(figsize=(15, 10))
    plt.scatter(V, peaks_height)
    plt.xlabel(r'Noise variance $\left[ \frac{K^2}{year} \right]$')
    plt.ylabel(r'Peak height $\left[ K^2 \cdot year \right]$')
    plt.title('Peak height as a function of the noise variance', fontsize=15, fontweight='bold')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=True))
    plt.savefig(peak_heights_plot_destination)

def plot_SNR():
    """
    Plot the signal-to-noise ratio as a function of the noise variance.

    This function generates a scatter plot illustrating the signal-to-noise ratio as a function of the noise variance.
    The signal-to-noise ratio is computed as the ratio between the strength of the peak and the baseline height.
    """

    variance_start = config['settings'].getfloat('variance_start')
    variance_end = config['settings'].getfloat('variance_end')
    num_variances = config['settings'].getint('num_variances')

    V = np.linspace(variance_start, variance_end, num=num_variances)

    SNR = check_path_and_load_data(data_path = SNR_path,
                                   contained_data = 'signal-to-noise ratio')
    
    f = plt.figure(figsize=(15, 10))
    plt.scatter(V, SNR)
    plt.xlabel(r'Noise variance $\left[ \frac{K^2}{year} \right]$')
    plt.ylabel(r'SNR')
    plt.title('Signal-to-noise ratio as a function of the noise variance', fontsize=15, fontweight='bold')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=True))
    plt.savefig(SNR_plot_destination)

def plot_pseudo_potential():

    data = check_path_and_load_data(data_path = pseudo_potential_plot_path,
                                    contained_data = 'pseudo-potential')  
    temperature = data[:, 0]
    potential_min_forcing = data[:, 1]
    potential_no_forcing = data[:, 2]
    potential_max_forcing = data[:, 3]

    f = plt.figure(figsize=(15, 10))
    plt.plot(temperature, potential_no_forcing, color='red', label=r'Periodic forcing value = $1$')
    plt.plot(temperature, potential_max_forcing, color='blue', label=r'Periodic forcing value = $1 + 10^{-4}$')
    plt.plot(temperature, potential_min_forcing, color='green', label=r'Periodic forcing value = $1 - 10^{-4}$')
    plt.legend(loc='best')
    plt.xlabel(r'Temperature $\left[ K \right]$')
    plt.ylabel(r'Pseudo-potential')
    plt.title('Pseudo-potential as a function of the temperature', fontsize=15, fontweight='bold')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    plt.savefig(pseudo_potential_plot_destination)  
    

plot_pseudo_potential()
peak_strenght_plot()
peak_height_plot()
plot_SNR()

with aes.green_text():
    print('Plots saved!')
