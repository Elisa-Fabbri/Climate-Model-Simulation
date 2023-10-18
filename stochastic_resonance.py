"""
This module contains the definition of the functions used in the stochastic resonance project.

In particular, it contains the following functions:
- emitted_radiation: Calculate the emitted radiation from the Earth's surface based on the 
  chosen emission model.
- periodic_forcing: Calculate the periodic forcing applied to the system.
- calculate_rate_of_temperature_change: Calculate the rate of temperature change (dT/dt) 
  given a certain temperature (T) using a specified emission model for the computation of the constant 
  beta.
- simulate_ito: Simulate temperature using the Itô stochastic differential equation.
- find_peak_indices: Find indices of theoretically predicted peaks in a frequency spectrum.
- calculate_peaks: Calculate the values of peaks in a power spectral density.
- calculate_peaks_base: Calculate the values of the base of the peaks in a power spectral density.
- calculate_peak_height: Calculate the heights of peaks in a power spectral density.
"""

import numpy as np
import aesthetics as aes
from scipy.integrate import quad


#Default values for the functions:

# Stable and unstable temperatures solutions in Kelvin:
stable_temperature_solution_1_default = 280
unstable_temperature_solution_default = 285
stable_temperature_solution_2_default = 290

surface_earth_thermal_capacity_j_per_m2_K = 0.31e9 #average earth surface thermal capacity [J/(m^2 K)]
relaxation_time_default = 13 #in years
emission_model_default = 'linear'

#periodic forcing constants:
forcing_amplitude_default = 0.0005
forcing_period_default = 1e5
forcing_angular_frequency_default = (2*np.pi)/forcing_period_default #computes the angular frequency 
                                                                     #of the periodic forcing

num_sec_in_a_year = 365.25*24*60*60
#the following line of code converts the earth thermal capacity from J/(m^2 K) to kg/(year^2 K)
surface_earth_thermal_capacity_in_years = surface_earth_thermal_capacity_j_per_m2_K*(num_sec_in_a_year**2)

#parameters for the simulation (used in the simulate ito function):
time_step_default = 1 # year
num_steps_default = 1000000 # years
num_simulations_default = 10

def emitted_radiation(temperature,
                      emission_model=emission_model_default,
                      conversion_factor=num_sec_in_a_year
                      ):
    """
    Calculate the emitted radiation from the Earth's surface based on the chosen emission model.

    This function calculates the radiation emitted from the Earth's surface based on the 
    specified emission model.
    The emission model can be either 'linear' or 'black body'. 
    The emitted radiation is computed using the given temperature and conversion factor.

    Parameters:
    - temperature (float): The temperature of the Earth's surface in Kelvin. It must be a non-negative value.
    - emission_model (str, optional): The emission model to use, which can be 'linear' or 'black body'.
      Default is 'linear'.
    - conversion_factor (float, optional): The conversion factor for time units to calculate the 
      emitted radiation, measured in kg/year^3. 
      The default is set to the number of seconds in a year.

    Returns:
    - emitted_radiation (float): The calculated emitted radiation in kg/year^3.

    Raises:
    - ValueError: If an invalid emission model is selected. Valid options are 'linear' or 'black body'.
    - ValueError: If the temperature is not a non-negative value.

    Notes:
    For an explanation on the linear model and on the values of the parameters A and B,
    see `<https://www.pnas.org/doi/10.1073/pnas.1809868115>`.
    """

    if np.any(temperature < 0):
        raise ValueError('Temperature in Kelvin must be non-negative.')
    
    if emission_model == 'linear':
        A = -339.647 * (conversion_factor ** 3)  # converts from W/(m^2 K) to kg/year^3 K
        B = 2.218 * (conversion_factor ** 3)  # converts from W/(m^2 K) to kg/year^3 K
        emitted_radiation = A + B * temperature
    elif emission_model == 'black body':
        Stefan_Boltzmann_constant = 5.67e-8 * (conversion_factor ** 3)  # W/(m^2 K^4)
        emitted_radiation = Stefan_Boltzmann_constant * (temperature ** 4)
    else:
        raise ValueError('Invalid emission model selection. Choose "linear" or "black body".')

    return emitted_radiation

def periodic_forcing(time,
                     amplitude = forcing_amplitude_default,
                     angular_frequency = forcing_angular_frequency_default
                     ):
    """
    Calculate the periodic forcing applied to the system.

    This function calculates the periodic forcing applied to a system at a given time or times. 
    The periodic forcing is modeled as an oscillatory function of time.

    Parameters:
    - time (float or array-like): The time or times at which to calculate the periodic forcing.
    - amplitude (float, optional): The amplitude of the periodic forcing. Default is 0.0005.
    - angular_frequency (float, optional): The angular frequency of the periodic forcing. 
      Default is (2 * pi) / 1e5.

    Returns:
    - periodic_forcing (float or array-like): The calculated periodic forcing values corresponding to 
      the input time(s).

    Raises:
    - ValueError: If the time is a negative value.
    """
    if np.any(np.array(time) < 0):
        raise ValueError('Time must be non-negative.')
    
    periodic_forcing = np.array(amplitude) * np.cos(np.array(angular_frequency) * time)
    return 1 + periodic_forcing

def calculate_rate_of_temperature_change(
                 temperature,
      					 surface_thermal_capacity = surface_earth_thermal_capacity_in_years,
      					 relaxation_time = relaxation_time_default,
      					 stable_temperature_solution_1 = stable_temperature_solution_1_default,
      					 unstable_temperature_solution = unstable_temperature_solution_default,
      					 stable_temperature_solution_2 = stable_temperature_solution_2_default,
      					 emission_model = emission_model_default,
      					 periodic_forcing_value = 1
                 ):

    """
    Calculate the rate of temperature change (dT/dt) given a certain temperature (T) using a 
    specified emission model for the computation of the constant beta.

    This function calculates the rate of temperature change (dT/dt) at a specified temperature (T) 
    using a specified emission model for the computation of the constant beta.

    Parameters:
    - temperature (float): The temperature in Kelvin at which to calculate the rate of 
      temperature change (dT/dt).
    - surface_thermal_capacity (float, optional): The surface thermal capacity in kg per square year per Kelvin. 
      Default corresponds to 0.31e9 joules per square meter per Kelvin.
    - relaxation_time (float, optional): The relaxation time in years. Default is 13.
    - stable_temperature_solution_1 (float, optional): The first stable temperature solution in Kelvin. 
      Default is 280.
    - unstable_temperature_solution (float, optional): The unstable temperature solution in Kelvin. 
      Default is 285.
    - stable_temperature_solution_2 (float, optional): The second stable temperature solution in Kelvin. 
      Default is 290.
    - emission_model (str, optional): The emission model to use for beta computation. 
      Default is 'linear'.
    - periodic_forcing_value (float, optional): The amplitude of the periodic forcing in the equation. 
      Default is 1.

    Returns:
    - beta (float): The calculated constant beta.
    - denominator (float): The denominator of the equation.
    - dT/dt (float): The calculated rate of temperature change (dT/dt) at the specified temperature (T).
    """

    beta = -((surface_thermal_capacity / (relaxation_time * emitted_radiation(
        temperature=stable_temperature_solution_2,
        emission_model=emission_model))) *
        ((stable_temperature_solution_1 * unstable_temperature_solution * stable_temperature_solution_2) /
        ((stable_temperature_solution_1 - stable_temperature_solution_2) *
         (unstable_temperature_solution - stable_temperature_solution_2))))
    
    denominator = 1 + beta * (1 - (temperature / stable_temperature_solution_1)) * (
        1 - (temperature / unstable_temperature_solution)) * (
        1 - (temperature / stable_temperature_solution_2))

    dT_dt = (emitted_radiation(temperature=temperature, emission_model=emission_model) /
        surface_thermal_capacity) * (
       (periodic_forcing_value /
        denominator) - 1)
    
    return beta, denominator, dT_dt

def pseudo_potential(
        upper_temperature_limit,
        lower_temperature_limit = stable_temperature_solution_1_default - 10,
        surface_thermal_capacity = surface_earth_thermal_capacity_in_years,
      	relaxation_time = relaxation_time_default,
      	stable_temperature_solution_1 = stable_temperature_solution_1_default,
      	unstable_temperature_solution = unstable_temperature_solution_default,
      	stable_temperature_solution_2 = stable_temperature_solution_2_default,
      	emission_model = emission_model_default,
      	periodic_forcing_value = 1
        ):
  """
  Calculate the pseudo-potential phi.

  This function calculates the pseudo-potential phi using the specified parameters.
  In order to calculate the pseudo-potential phi, the rate of temperature change (dT/dt) is calculated
  using the calculate_rate_of_temperature_change function and then integrated over the specified range.

  Parameters:
  - upper_temperature_limit (float): The upper limit of the integral.
  - lower_temperature_limit (float, optional): The lower limit of the integral. Default is 10 degrees below
    the minimum temperature (stable_temperature_solution_1).
  - surface_thermal_capacity (float, optional): The surface thermal capacity in kg per square year per Kelvin.
    Default corresponds to 0.31e9 joules per square meter per Kelvin.
  - relaxation_time (float, optional): The relaxation time in years. Default is 13.
  - stable_temperature_solution_1 (float, optional): The first stable temperature solution in Kelvin.
    Default is 280.
  - unstable_temperature_solution (float, optional): The unstable temperature solution in Kelvin.
    Default is 285.
  - stable_temperature_solution_2 (float, optional): The second stable temperature solution in Kelvin.
    Default is 290.
  - emission_model (str, optional): The emission model to use for beta computation.
    Default is 'linear'.
  - periodic_forcing_value (float, optional): The amplitude of the periodic forcing in the equation.
    Default is 1.
  
  Returns:
  - phi (float): The calculated pseudo-potential phi.
  - error (float): An estimate of the absolute error in the integration.
  """
  result, error =  quad(lambda temperature: 
                       calculate_rate_of_temperature_change(temperature = temperature,
      					       surface_thermal_capacity = surface_thermal_capacity,
      					       relaxation_time = relaxation_time,
      					       stable_temperature_solution_1 = stable_temperature_solution_1,
      					       unstable_temperature_solution = unstable_temperature_solution,
      					       stable_temperature_solution_2 = stable_temperature_solution_2,
      					       emission_model = emission_model,
      					       periodic_forcing_value = periodic_forcing_value)[2], 
                       lower_temperature_limit, upper_temperature_limit)
  phi = -result
  return phi, error

def Kramer_rate(noise_variance,
                potential_second_derivative_in_minima = 2, 
                potential_second_derivative_in_maxima = -1,
                barrier_height = 0.25
                ):
    
    """
    Calculate the Kramer rate.

    This function calculates the Kramer rate using the specified parameters.

    Parameters:
    - noise_variance (float): The variance of the noise.
    - potential_second_derivative_in_minima (float, optional): The second derivative of the potential in the minima.
      Default is 2.
    - potential_second_derivative_in_maxima (float, optional): The second derivative of the potential in the maxima.
      Default is -1.
    - barrier_height (float, optional): The barrier height. Default is 0.25.

    Returns:
    - r_k (float): The calculated Kramer rate.
    """
    
    r_k = np.sqrt(potential_second_derivative_in_minima * \
                  (- potential_second_derivative_in_maxima)) / (2 * np.pi) * \
                  np.exp(- barrier_height / noise_variance)
    
    return r_k

def trascendental_equation_for_noise_variance(noise_variance,
                                              potential_second_derivative_in_minima = 2,
                                              potential_second_derivative_in_maxima = -1,
                                              barrier_height = 0.25,
                                              forcing_angular_frequency = forcing_angular_frequency_default):
    
    """
    Calculate the value of the trascendental equation for the noise variance.

    This function calculates the value of the trascendental equation for the noise variance using the specified parameters.

    Parameters:
    - noise_variance (float): The variance of the noise.
    - potential_second_derivative_in_minima (float, optional): The second derivative of the potential in the minima.
      Default is 2.
    - potential_second_derivative_in_maxima (float, optional): The second derivative of the potential in the maxima.
      Default is -1.
    - barrier_height (float, optional): The barrier height. Default is 0.25.
    - forcing_angular_frequency (float, optional): The angular frequency of the periodic forcing.
      Default is (2 * pi) / 1e5.

    Returns:
    - equation_value (float): The calculated trascendental equation for the noise variance.
    """
    
    equation_value = 4*(Kramer_rate(noise_variance = noise_variance,
                                    potential_second_derivative_in_minima = potential_second_derivative_in_minima,
                                    potential_second_derivative_in_maxima = potential_second_derivative_in_maxima,
                                    barrier_height = barrier_height)**2) - \
                                    (forcing_angular_frequency**2 * \
                                    ((barrier_height/noise_variance) - 1))
    return equation_value

def time_scale_matching_condition(barrier_height = 0.25,
                                  forcing_period = forcing_period_default,
                                  potential_second_derivative_in_minima = 2,
                                  potential_second_derivative_in_maxima = -1):
    """
    This function calculates the value of the noise intensity that satisfies the time scale matching condition.

    This function calculates the value of the noise intensity that satisfies the time scale matching condition
    using the specified parameters.

    Parameters:
    - barrier_height (float, optional): The barrier height. Default is 0.25.
    - forcing_period (float, optional): The period of the periodic forcing. Default is 1e5.
    - potential_second_derivative_in_minima (float, optional): The second derivative of the potential in the minima.
      Default is 2.
    - potential_second_derivative_in_maxima (float, optional): The second derivative of the potential in the maxima.
      Default is -1.
    
    Returns:
    - noise_variance (float): The calculated noise variance that satisfies the time scale matching condition.
    """
    
    log_argument = 4*(np.pi)*(1/forcing_period)* \
      (1/(np.sqrt(potential_second_derivative_in_minima*(- potential_second_derivative_in_maxima)))) 
    D_SR_approx = - barrier_height/np.log(log_argument) 
    return D_SR_approx

def simulate_ito(
    T_start = stable_temperature_solution_2_default,
    t_start=0,
    noise_variance = 0,
    dt = time_step_default,
    num_steps = num_steps_default,
    num_simulations = num_simulations_default,
    surface_thermal_capacity = surface_earth_thermal_capacity_in_years,
    relaxation_time = relaxation_time_default,
    stable_temperature_solution_1 = stable_temperature_solution_1_default,
    unstable_temperature_solution = unstable_temperature_solution_default,
    stable_temperature_solution_2 = stable_temperature_solution_2_default,
    forcing_amplitude = forcing_amplitude_default,
    forcing_angular_frequency = forcing_angular_frequency_default,
    noise = True,
    emission_model = emission_model_default,
    forcing = 'varying',
    seed_value = 0
    ):

    """
    Simulate temperature using the Itô stochastic differential equation.

    This function simulates temperature evolution using the Itô stochastic differential equation (SDE).
    It allows for the modeling of temperature dynamics with various parameters, noise, and forcing.

    Parameters:
    - T_start (float): The initial temperature value at t_start. Default is 290.
    - t_start (float): The starting time of the simulation. Default is 0.
    - noise_variance (float): The variance of the noise. Default is 0 (no noise).
    - dt (float): The time step size for the simulation in years. Default is 1 year.
    - num_steps (int): The number of time steps in the simulation. Default is 1000000.
    - num_simulations (int): The number of simulation runs. Default is 10.
    - surface_thermal_capacity (float): The thermal capacity of the Earth's surface in kg/(year^2 K).
    Default is the average Earth surface thermal capacity in kg/(year^2 K), 
    which corresponds to 0.31e9 J/(m^2 K).
    - relaxation_time (int): The relaxation time in years. Default is 13 years.
    - stable_temperature_solution_1 (float): The stable temperature solution 1 in Kelvin.
    Default is 280 K.
    - unstable_temperature_solution (float): The unstable temperature solution in Kelvin.
    Default is 285 K.
    - stable_temperature_solution_2 (float): The stable temperature solution 2 in Kelvin.
    Default is 290 K.
    - forcing_amplitude (float): The amplitude of periodic forcing. Default is 0.0005.
    - forcing_angular_frequency (float): The angular frequency of periodic forcing.
    Default is (2 * pi) / 1e5.
    - noise (bool): A flag indicating whether to include noise in the simulation. Default is True.
    - emission_model (str): The emission model to use ('linear' or 'black body'). Default is 'linear'.
    - forcing (str): The type of forcing to apply ('constant' or 'varying'). Default is 'varying'.
    - seed_value (int): The seed value for the random number generator. Default is 0.

    Returns:
    - t (numpy.ndarray): An array of time values for the simulation.
    - T (numpy.ndarray): An array of temperature values for each simulation run and time step.
    """
    
    np.random.seed(seed_value)
    sigma = np.sqrt(noise_variance)
    t = np.arange(t_start, t_start + num_steps * dt, dt)  # len(t) = num_steps
    T = np.zeros((num_simulations, num_steps))
    T[:, 0] = T_start

    if noise == True:
        W = np.random.normal(0, np.sqrt(dt), (num_simulations, num_steps))
    elif noise == False:
        W = np.zeros((num_simulations, num_steps))
    else:
        raise ValueError("Invalid value for 'noise'. Please use True or False")
    
    if forcing == "constant":
        forcing_values = np.ones(num_steps)
    elif forcing == "varying":
        forcing_values = periodic_forcing(time = t, amplitude = forcing_amplitude, 
                                          angular_frequency = forcing_angular_frequency)
    else:
        raise ValueError("Invalid value for 'forcing'. Please use 'constant' or 'varying'")
    
    for i in aes.progress(range(num_steps - 1)):
        Fi = calculate_rate_of_temperature_change(
            temperature = T[:, i],
            surface_thermal_capacity = surface_thermal_capacity,
            relaxation_time = relaxation_time,
            stable_temperature_solution_1 = stable_temperature_solution_1,
            unstable_temperature_solution = unstable_temperature_solution,
            stable_temperature_solution_2 = stable_temperature_solution_2,
            emission_model = emission_model,
            periodic_forcing_value = forcing_values[i]
        )[2]
        dT = Fi * dt + sigma * W[:, i]
        T[:, i + 1] = T[:, i] + dT

    return t, T

def binarize_temperature_inplace(temperature,
                                 stable_temperature_solution_1=stable_temperature_solution_1_default,
                                 stable_temperature_solution_2=stable_temperature_solution_2_default):
    """
    Binarize temperature values in-place.

    This function binarizes temperature values based on the given stable temperature solutions, 
    modifying the input array in-place.

    Parameters:
    - temperature (array-like): An array of temperature values.
    - stable_temperature_solution_1 (float): The stable temperature solution 1 in Kelvin.
      Default is 280 K.
    - stable_temperature_solution_2 (float): The stable temperature solution 2 in Kelvin.
      Default is 290 K.
    """
    threshold = (stable_temperature_solution_1 + stable_temperature_solution_2) / 2
        
    temperature[temperature < threshold] = stable_temperature_solution_1
    temperature[temperature >= threshold] = stable_temperature_solution_2

def find_peak_indices(frequencies, 
                      period = forcing_period_default
                      ):
    """
    Find indices of theoretically predicted peaks in a frequency spectrum.

    This function calculates the indices of the theoretically predicted peaks in a frequency 
    spectrum based on the specified period.

    Parameters:
    - frequencies (array-like): An array of frequencies.
    - period (float): The period used for peak prediction which should be the period of the periodic 
    forcing applied to the system. Default is 1e5.

    Returns:
    - peaks_indices (array): An array of indices corresponding to the closest frequencies to the 
      predicted peak frequencies based on the given period.
    """
    peaks_indices = np.abs(frequencies - (1/period)).argmin()
    return peaks_indices

def calculate_peaks(PSD_mean, peak_index):
    """
    Calculate the values of peaks in a power spectral density.

    This function calculates the values of peaks in a power spectral density (PSD) based on the 
    provided PSD_mean and a single peak index.

    Parameters:
    - PSD_mean (array-like): An array of mean power spectral density values.
    - peak_index (int): An integer representing the index of the peak position.

    Returns:
    - peak (float): The value of the peak corresponding to the provided peak index.
    """
    peak = PSD_mean[peak_index]
    return peak


def calculate_peaks_base(PSD_mean, peak_index, num_neighbors=2):
    """
    Calculate the values of the base of the peaks in a power spectral density.

    This function calculates the values of the base of the peaks in a power spectral density (PSD) based on 
    the provided PSD_mean, peak index, and the number of neighboring points to consider for 
    the base calculation.

    Parameters:
    - PSD_mean (array-like): An array of mean power spectral density values.
    - peak_index (int): An integer representing the index of the peak position.
    - num_neighbors (int, optional): The number of neighboring points on each side of a peak to consider 
      for calculating the peak's base. Default is 2.

    Returns:
    - peak_base (float): The value of the base of the peak corresponding to the provided peak index.
    """
    current_index = peak_index
    neighbor_indices = np.arange(current_index - num_neighbors, 
                                 current_index + num_neighbors + 1)
    valid_indices = np.clip(neighbor_indices, 0, len(PSD_mean) - 1)
    valid_indices = valid_indices[valid_indices != current_index]
    valid_indices = np.unique(valid_indices)
    neighbor_values = PSD_mean[valid_indices]
    peak_base = np.mean(neighbor_values)

    return peak_base


def calculate_peak_height(peak, 
                          peak_base
                          ):
    """
    Calculate the heights of peaks in a power spectral density.

    This function calculates the heights of peaks in a power spectral density (PSD) based on the provided 
    peak values and peak base values.

    Parameters:
    - peaks (array-like): An array of peak values.
    - peaks_base (array-like): An array of base values corresponding to the peaks.

    Returns:
    - peak_height (array): An array of peak heights calculated as the difference between peak values and 
    peak base values.
    """
    peak_height = peak - peak_base
    return peak_height

def calculate_SNR(peak, peak_base):
    """
    Calculate the signal-to-noise ratio (SNR) of peaks in a power spectral density.

    This function calculates the signal-to-noise ratio (SNR) of peaks in a power spectral density (PSD)
    based on the provided peak values and peak base values. If both peaks and peaks_base are zero, SNR is set to 0.
    If only one of peaks and peaks_base is zero, SNR is set to 0 and a warning is printed.

    Parameters:
    - peaks (array-like): An array of peak values.
    - peaks_base (array-like): An array of base values corresponding to the peaks.

    Returns:
    - SNR (array): An array of SNR values calculated as the ratio between peak heights and peak base values, or 0 if both peaks and peaks_base are zero.
    """

    if (peak != 0) and (peak_base != 0):
        SNR = 2 * peak / peak_base
    elif (peak == 0) and (peak_base == 0):
        SNR = 0
    else:
        SNR = 0
        with aes.orange_text():
          print(f"Warning: Unexpected behavior for peaks = {peak} and peaks_base = {peak_base}. \
                SNR set to zero")
          
    return SNR
