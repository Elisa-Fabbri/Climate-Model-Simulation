import numpy as np
import stochastic_resonance as sr
import configparser
import sys
import os
import aesthetics as aes
from scipy.optimize import newton
from scipy.misc import derivative

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

os.makedirs('data', exist_ok = True)

pseudo_potential_destination = config['data_paths'].get('pseudo_potential')

#----Computing the singular point of F(T), the deterministic rate of temperature change-----

print('Computing the singular point of F(T), the deterministic rate of temperature change...')

# Newton-Raphson method for solving the trascendent equation denominator (of the rate of temperature change) = 0:

initial_guess = 50
singular_point = newton(lambda temperature: sr.calculate_rate_of_temperature_change(
                        temperature = temperature,
                        surface_thermal_capacity=C_years,
                        relaxation_time=relaxation_time,
                        stable_temperature_solution_1=stable_temperature_solution_1,
                        unstable_temperature_solution=unstable_temperature_solution,
                        stable_temperature_solution_2=stable_temperature_solution_2,
                        emission_model=emission_model)[1], initial_guess)

print("Singular point found at a temperature value of ", singular_point, 'Kelvin')

#----Computing the pseudo-potential-----

print('Computing the value of the pseudo-potential in the critical points (stable and unstable solutions)...')
with aes.orange_text():
    print('Note that the value of the potential is obtained by integrating the rate of temperature change from the right of the singular point')

steady_temperatures_solutions = [stable_temperature_solution_1,
                                 unstable_temperature_solution,
                                 stable_temperature_solution_2]

potential_values = np.zeros((len(steady_temperatures_solutions), 2))

for index, temperature in enumerate(steady_temperatures_solutions):
    potential_value, error = sr.pseudo_potential(
                                         upper_temperature_limit = temperature,
                                         lower_temperature_limit = singular_point + 0.00001, 
                                         surface_thermal_capacity = C_years,
                                         relaxation_time = relaxation_time,
                                         stable_temperature_solution_1 = stable_temperature_solution_1,
                                         unstable_temperature_solution = unstable_temperature_solution,
                                         stable_temperature_solution_2 = stable_temperature_solution_2,
                                         emission_model = emission_model)
    potential_values[index, 0] = potential_value
    potential_values[index, 1] = error

print(' Temperature (K) | Pseudo-potential | Error ')
print('----------------------------------------------------')
print(' {:^15.2f} | {:^20.4f} | {:^10.4f} '.format(stable_temperature_solution_1, 
                                                   potential_values[0, 0], 
                                                   potential_values[0, 1]))
print(' {:^15.2f} | {:^20.4f} | {:^10.4f} '.format(unstable_temperature_solution,
                                                    potential_values[1, 0], 
                                                    potential_values[1, 1]))
print(' {:^15.2f} | {:^20.4f} | {:^10.4f} '.format(stable_temperature_solution_2,
                                                    potential_values[2, 0],
                                                    potential_values[2, 1]))
print('----------------------------------------------------')

#----Computing the barrier height-----

barrier_height_left = potential_values[1, 0] - potential_values[0, 0]
barrier_height_right = potential_values[1, 0] - potential_values[2, 0]

barrier_height = 0.5*(barrier_height_left + barrier_height_right)
barrier_height_error = 0.5*(barrier_height_right - barrier_height_left)

print('The barrier height is approximatively ', barrier_height , ' +/- ', barrier_height_error)

#----Computing the second derivative of the potential in the steady states-----

print('Computing the second derivative of the potential in the steady states...')

potential_second_derivative = np.zeros((len(steady_temperatures_solutions), 2))

dx = 1e-5
for index, steady_temperature in enumerate(steady_temperatures_solutions):
    second_derivative_value = - derivative(lambda temperature: sr.calculate_rate_of_temperature_change(
                                temperature = temperature,
                                surface_thermal_capacity=C_years,
                                relaxation_time=relaxation_time,
                                stable_temperature_solution_1=stable_temperature_solution_1,
                                unstable_temperature_solution=unstable_temperature_solution,
                                stable_temperature_solution_2=stable_temperature_solution_2,
                                emission_model=emission_model)[2], 
                                steady_temperature, 
                                dx=dx)
    potential_second_derivative[index, 0] = second_derivative_value
    potential_second_derivative[index, 1] = dx

print(' Temperature (K) | Second derivative | Error ')
print('--------------------------------------------------------------')
print(' {:^15.2f} | {:^25.4f} | {:^10.4f} '.format(stable_temperature_solution_1, 
                                                   potential_second_derivative[0, 0], 
                                                   potential_second_derivative[0, 1]))
print(' {:^15.2f} | {:^25.4f} | {:^10.4f} '.format(unstable_temperature_solution,
                                                    potential_second_derivative[1, 0], 
                                                    potential_second_derivative[1, 1]))
print(' {:^15.2f} | {:^25.4f} | {:^10.4f} '.format(stable_temperature_solution_2,
                                                    potential_second_derivative[2, 0],
                                                    potential_second_derivative[2, 1]))

potential_second_derivative_in_maxima = potential_second_derivative[1, 0]
potential_second_derivative_in_minima_left = potential_second_derivative[0, 0]
potential_second_derivative_in_minima_right = potential_second_derivative[2, 0]

potential_second_derivative_in_minima = 0.5*(potential_second_derivative_in_minima_left + 
                                             potential_second_derivative_in_minima_right)

potential_second_derivative_in_minima_error = 0.5*(potential_second_derivative_in_minima_right -
                                                   potential_second_derivative_in_minima_left)

print('The value of the second derivative of the potential in minima is ', potential_second_derivative_in_minima,
      ' +/- ', potential_second_derivative_in_minima_error)

print('--------------------------------------------------------------')

# Newton-Raphson method for solving the trascendent equation for finding D_SR:

initial_guess = 0.1

D_SR = newton(lambda noise_variance: sr.trascendental_equation_for_noise_variance(
              noise_variance = noise_variance,
              potential_second_derivative_in_minima= potential_second_derivative_in_minima,
              potential_second_derivative_in_maxima= potential_second_derivative_in_maxima,
              barrier_height = barrier_height,
              forcing_angular_frequency=forcing_angular_frequency),
              initial_guess)

D_SR_apprx = sr.time_scale_matching_condition(barrier_height=barrier_height,
                                              forcing_period=forcing_period,
                                              potential_second_derivative_in_maxima=potential_second_derivative_in_maxima,
                                              potential_second_derivative_in_minima=potential_second_derivative_in_minima)

log_argument = (4*np.pi)/( forcing_period*np.sqrt((-potential_second_derivative_in_maxima)*
                                                  potential_second_derivative_in_minima))
partial_derivative_D_SR_wrt_barrier_height = - (1 / np.log(log_argument))

partial_derivative_D_SR_wrt_potential_second_derivative_in_mimina = - ( barrier_height /
                                                                       (2*potential_second_derivative_in_minima*\
                                                                        np.power(np.log(log_argument), 2)))

D_SR_apprx_error = np.sqrt((partial_derivative_D_SR_wrt_barrier_height**2)*(barrier_height_error**2) +
                            (partial_derivative_D_SR_wrt_potential_second_derivative_in_mimina**2)*\
                            (potential_second_derivative_in_minima_error**2))

print('The value of the noise variance that maximizes the the response amplitude should be around ', 
      D_SR, '+/-', D_SR_apprx_error)
print('The approximated value for the noise variance that maximizes the the response amplitude is ', D_SR_apprx,
      ' +/- ', D_SR_apprx_error)
print('The value of the noise variance that maximizes the signal-to-noise ratio should be around ', 
      barrier_height/2, '+/-', barrier_height_error/2)

#------------Generation of the data for the pseudo potential plot----------------

print('Generating data for the pseudo potential plot...')

temperature = np.linspace(stable_temperature_solution_1 - 5,
                          stable_temperature_solution_2 + 5, 1000)

pseudo_potential_data = np.zeros((len(temperature), 3))

for index, temp in enumerate(temperature):
    potential, error = sr.pseudo_potential(upper_temperature_limit = temp,
                                           lower_temperature_limit = singular_point + 0.00001, 
                                           surface_thermal_capacity = C_years,
                                           relaxation_time = relaxation_time,
                                           stable_temperature_solution_1 = stable_temperature_solution_1,
                                           unstable_temperature_solution = unstable_temperature_solution,
                                           stable_temperature_solution_2 = stable_temperature_solution_2,
                                           emission_model = emission_model)
    pseudo_potential_data[index, 0] = temp
    pseudo_potential_data[index, 1] = potential
    pseudo_potential_data[index, 2] = error

np.save(pseudo_potential_destination, pseudo_potential_data)

with aes.green_text():
    print('Result saved!')