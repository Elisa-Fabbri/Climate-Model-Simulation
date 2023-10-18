"""
This module contains the tests for the functions in the stochastic_resonance module.

The tests are organized into sections with comments, each focusing on a specific function within the 
stochastic_resonance module.
"""

import numpy as np
import stochastic_resonance as sr
import pytest
import random as rn

#Test emitted_radiation_function in the linear case 

steady_temperature_solutions_default_linear_test = [
    (sr.stable_temperature_solution_1_default, -339.647 + 2.218*280), # glacial temperature
    (sr.unstable_temperature_solution_default, 
     -339.647 + 2.218*sr.unstable_temperature_solution_default), # ustable temperature solution
    (sr.stable_temperature_solution_2_default, 
     -339.647 + 2.218*sr.stable_temperature_solution_2_default), # interglacial temperature
]

@pytest.mark.parametrize("steady_temperature, expected_radiation", 
                         steady_temperature_solutions_default_linear_test)
def test_emitted_radiation_linear(steady_temperature, expected_radiation):
    """
    Test the linear emitted radiation in W/m^2 for temperature solutions default parameters.

    GIVEN: a steady temperature and the expected emitted radiation in K
    WHEN: the emitted_radiation function is called with conversion_factor = 1
    THEN: the result should be the expected emitted radiation in W/m^2
    """
    
    expected_value = expected_radiation
    calculated_value = sr.emitted_radiation(steady_temperature, 
                                            emission_model='linear', conversion_factor=1)
    
    assert calculated_value == pytest.approx(expected_value, rel=1e-6)

@pytest.mark.parametrize("steady_temperature, expected_radiation_W_m2", 
                         steady_temperature_solutions_default_linear_test)
def test_emitted_radiation_conversion(steady_temperature, expected_radiation_W_m2):
    """
    Test the linear emitted radiation in kg/m^3 for temperature solutions default parameters.

    GIVEN: a steady temperature and the expected emitted radiation in K
    WHEN: the emitted_radiation function is called with conversion_factor = num_sec_in_a_year
    THEN: the result should be the expected emitted radiation in kg/year^3
    """
    
    expected_value = expected_radiation_W_m2*(sr.num_sec_in_a_year**3)
    calculated_value = sr.emitted_radiation(steady_temperature, emission_model='linear', 
                                            conversion_factor=sr.num_sec_in_a_year)
    
    assert calculated_value == pytest.approx(expected_value, rel=1)

#Test emitted_radiation_function in the black body case

steady_temperature_solutions_default_black_body_test = [
    (sr.stable_temperature_solution_1_default, 
     5.67e-8*sr.stable_temperature_solution_1_default**4), # glacial temperature
    (sr.unstable_temperature_solution_default, 
     5.67e-8*sr.unstable_temperature_solution_default**4), # ustable temperature solution
    (sr.stable_temperature_solution_2_default, 
     5.67e-8*sr.stable_temperature_solution_2_default**4), # interglacial temperature
]

@pytest.mark.parametrize("steady_temperature, expected_radiation", 
                         steady_temperature_solutions_default_black_body_test)
def test_emitted_radiation_black_body(steady_temperature, expected_radiation):
    """
    Test the black body emitted radiation in W/m^2 for temperature solutions default parameters.

    GIVEN: a steady temperature and the expected emitted radiation in K
    WHEN: the emitted_radiation function is called with conversion_factor = 1
    THEN: the result should be the expected emitted radiation in W/m^2
    """
    
    expected_value = expected_radiation
    calculated_value = sr.emitted_radiation(steady_temperature, 
                                            emission_model='black body', 
                                            conversion_factor=1)
    
    assert calculated_value == pytest.approx(expected_value, rel=1e-6)

@pytest.mark.parametrize("steady_temperature, expected_radiation_W_m2", 
                         steady_temperature_solutions_default_black_body_test)
def test_emitted_radiation_conversion_black_body(steady_temperature, expected_radiation_W_m2):
    """
    Test the black body emitted radiation in kg/m^3 for temperature solutions default parameters.

    GIVEN: a steady temperature and the expected emitted radiation in K
    WHEN: the emitted_radiation function is called with conversion_factor = num_sec_in_a_year
    THEN: the result should be the expected emitted radiation in kg/year^3
    """
    
    expected_value = expected_radiation_W_m2*(sr.num_sec_in_a_year**3)
    calculated_value = sr.emitted_radiation(steady_temperature, 
                                            emission_model='black body', 
                                            conversion_factor=sr.num_sec_in_a_year)
    
    assert calculated_value == pytest.approx(expected_value, rel=1)

# Test the emitted_radiation function for invalid emission_model selection

def test_invalid_emitted_radiation_model():
    """Test the emitted_radiation function for an invalid emission model selection.
    
    GIVEN: an invalid emission model selection
    WHEN: the emitted_radiation function is called
    THEN: a ValueError should be raised
    """
    with pytest.raises(ValueError, 
                       match='Invalid emission model selection. Choose "linear" or "black body".'):
        sr.emitted_radiation(0, emission_model='invalid', conversion_factor=1)

    with pytest.raises(ValueError, 
                       match='Invalid emission model selection. Choose "linear" or "black body".'):
        sr.emitted_radiation(0, emission_model='invalid', conversion_factor=sr.num_sec_in_a_year)

#Test the emitted_radiation for negative negative temperature value

def test_invalid_emitted_radiation_temperature():
    """Test the emitted_radiation function for a negative temperature value.
    
    GIVEN: a negative temperature value
    WHEN: the emitted_radiation function is called
    THEN: a ValueError should be raised
    """
    with pytest.raises(ValueError, match='Temperature in Kelvin must be non-negative.'):
        sr.emitted_radiation(-1, emission_model='linear', conversion_factor=1)

    with pytest.raises(ValueError, match='Temperature in Kelvin must be non-negative.'):
        sr.emitted_radiation(-1, emission_model='linear', conversion_factor=sr.num_sec_in_a_year)

    with pytest.raises(ValueError, match='Temperature in Kelvin must be non-negative.'):
        sr.emitted_radiation(-1, emission_model='black body', conversion_factor=1)

    with pytest.raises(ValueError, match='Temperature in Kelvin must be non-negative.'):
        sr.emitted_radiation(-1, emission_model='black body', conversion_factor=sr.num_sec_in_a_year)


# Test periodic_forcing function

def test_periodic_forcing_max():
    """
    Test that the periodic forcing function returns the maximum value of the forcing amplitude 
    plus one when the time is equal to the forcing period.

    GIVEN: time equal to the forcing period and the forcing amplitude
    WHEN: the periodic_forcing function is called
    THEN: the result should be the maximum value of the forcing amplitude plus one
    """
    expected_value = 1 + sr.forcing_amplitude_default
    calculated_value = sr.periodic_forcing(sr.forcing_period_default, sr.forcing_amplitude_default)
    assert calculated_value == pytest.approx(expected_value, rel=1e-6)

def test_periodic_forcing_min():
    """
    Test that the periodic forcing function returns one minus the value of the forcing amplitude 
    when the time is equal to the forcing period divided by two.

    GIVEN: time equal to the forcing period and the forcing amplitude
    WHEN: the periodic_forcing function is called
    THEN: the result should be one minus the value of the forcing amplitude
    """
    expected_value = 1 - sr.forcing_amplitude_default
    calculated_value = sr.periodic_forcing(sr.forcing_period_default/2, sr.forcing_amplitude_default)
    assert calculated_value == pytest.approx(expected_value, rel=1e-6)

def test_periodic_forcing_is_one():
    """
    Test that the periodic forcing function returns one when the time is equal to the forcing period 
    divided by four.

    GIVEN: time equal to the forcing period divided by four and the forcing amplitude
    WHEN: the periodic_forcing function is called
    THEN: the result should be one
    """
    expected_value = 1
    calculated_value = sr.periodic_forcing(sr.forcing_period_default/4, sr.forcing_amplitude_default)
    assert calculated_value == expected_value
    
def test_periodic_forcing_for_list():
    """
    Test that the periodic forcing function returns a list of values when the time is a list of values.

    GIVEN: time is a list of values and the forcing amplitude
    WHEN: the periodic_forcing function is called
    THEN: the result should be a list of values containig the ordered periodic forcing values
    """
    expected_value = [1 - sr.forcing_amplitude_default, 1, 1 + sr.forcing_amplitude_default]
    calculated_value = sr.periodic_forcing([sr.forcing_period_default/2, 
                                            sr.forcing_period_default/4, 
                                            sr.forcing_period_default], 
                                            sr.forcing_amplitude_default)
    assert np.array_equal(calculated_value, expected_value)

def test_periodic_forcing_no_amplitude():
    """ 
    Test that the periodic forcing function returns one when the forcing amplitude is zero.

    GIVEN: a list of values for the time and a forcing amplitude of zero
    WHEN: the periodic_forcing function is called
    THEN: the result should be a list of ones
    """
    expected_value = [1, 1, 1]
    calculated_value = sr.periodic_forcing([0, sr.forcing_period_default, 2], 0)
    assert np.array_equal(calculated_value, expected_value)


# Test calculate_rate_of_temperature_change function

steady_temperature_solutions = [
    (sr.stable_temperature_solution_1_default),
    (sr.unstable_temperature_solution_default),
    (sr.stable_temperature_solution_2_default)
]

@pytest.mark.parametrize("steady_temperature", steady_temperature_solutions)
def test_calculate_rate_of_temperature_change_is_zero(steady_temperature):
    """ 
    Test that the rate of temperature change is zero for the steady temperature solutions. 

    GIVEN: a steady temperature
    WHEN: the calculate_rate_of_temperature_change function is called
    THEN: the result should be zero
    """
    expected_value = 0
    calculated_value = sr.calculate_rate_of_temperature_change(steady_temperature)[2]
    assert calculated_value == expected_value

def test_calculate_rate_of_temperature_change_stable_1():
    """
    Test the rate of temperature change near the first stable solution.

    GIVEN: a temperature value close to the first stable solution
    WHEN: the calculate_rate_of_temperature_change function is called
    THEN: the result should be negative if the input temperature is greater than the first stable solution,
            and positive if the input temperature is less than the first stable solution.
    """
    epsilon = 1  # small value to test near the stable solution

    assert sr.calculate_rate_of_temperature_change(sr.stable_temperature_solution_1_default + epsilon)[2] < 0
    assert sr.calculate_rate_of_temperature_change(sr.stable_temperature_solution_2_default - epsilon)[2] > 0

def test_calculate_rate_of_temperature_change_stable_2():
    """
    Test the rate of temperature change near the second stable solution.

    GIVEN: a temperature value close to the second stable solution
    WHEN: the calculate_rate_of_temperature_change function is called
    THEN: the result should be negative if the input temperature is greater than the second stable solution,
            and positive if the input temperature is less than the second stable solution.
    """
    epsilon = 1  # small value to test near the stable solution

    assert sr.calculate_rate_of_temperature_change(sr.stable_temperature_solution_2_default + epsilon)[2] < 0
    assert sr.calculate_rate_of_temperature_change(sr.stable_temperature_solution_2_default - epsilon)[2] > 0

def test_calculate_rate_of_temperature_change_unstable():
    """
    Test the rate of temperature change near the unstable solution.

    GIVEN: a temperature value close to the unstable solution
    WHEN: the calculate_rate_of_temperature_change function is called
    THEN: the result should be positive if the input temperature is greater than the unstable solution,
          and negative if the input temperature is less than the unstable solution.
    """
    epsilon = 1  # small value to test near the unstable solution

    assert sr.calculate_rate_of_temperature_change(sr.unstable_temperature_solution_default + epsilon)[2] > 0
    assert sr.calculate_rate_of_temperature_change(sr.unstable_temperature_solution_default - epsilon)[2] < 0

# Test pseudo_potential function

def test_pseudo_potential_for_stable_solutions_is_similar():
    """
    Test if the value of the pseudo-potential for the two stable solutions is approximately the same.

    GIVEN: the default parameters
    WHEN: the pseudo_potential function is called
    THEN: the value of the pseudo-potential for the two stable solutions should be approximately the same
    """
    potential_minima_1 = sr.pseudo_potential(sr.stable_temperature_solution_1_default)[0]
    potential_minima_2 = sr.pseudo_potential(sr.stable_temperature_solution_2_default)[0]
    assert potential_minima_1 == pytest.approx(potential_minima_2, rel=0.1)

def test_pseudo_potential_unstable_solution_greater_than_stable_solutions():
    """
    Test if the value of the pseudo-potential for the unstable solution is greater than the value of
    the pseudo-potential for the two stable solutions.

    GIVEN: the default parameters
    WHEN: the pseudo_potential function is called
    THEN: the value of the pseudo-potential for the unstable solution should be greater than the value of
          the pseudo-potential for the two stable solutions
    """

    potential_minima_1 = sr.pseudo_potential(sr.stable_temperature_solution_1_default)[0]
    potential_minima_2 = sr.pseudo_potential(sr.stable_temperature_solution_2_default)[0]
    potential_maxima = sr.pseudo_potential(sr.unstable_temperature_solution_default)[0]
    assert (potential_maxima > potential_minima_1 and potential_maxima > potential_minima_2)
    
def test_pseudo_potential_first_minima():
    """
    Test if the first stable temperature solution is a minima of the pseudo-potential.

    GIVEN: the default parameters
    WHEN: the pseudo_potential function is called
    THEN: the first stable temperature solution should be a minima of the pseudo-potential
    """
    epsilon = 1
    potential_minima_left = sr.pseudo_potential(sr.stable_temperature_solution_1_default - epsilon)[0]
    potential_minima = sr.pseudo_potential(sr.stable_temperature_solution_1_default)[0]
    potential_minima_right = sr.pseudo_potential(sr.stable_temperature_solution_1_default + epsilon)[0]

    assert (potential_minima_left > potential_minima and potential_minima_right > potential_minima)

def test_pseudo_potential_second_minima():
    """
    Test if the second stable temperature solution is a minima of the pseudo-potential.

    GIVEN: the default parameters
    WHEN: the pseudo_potential function is called
    THEN: the second stable temperature solution should be a minima of the pseudo-potential
    """
    epsilon = 1
    potential_minima_left = sr.pseudo_potential(sr.stable_temperature_solution_2_default - epsilon)[0]
    potential_minima = sr.pseudo_potential(sr.stable_temperature_solution_2_default)[0]
    potential_minima_right = sr.pseudo_potential(sr.stable_temperature_solution_2_default + epsilon)[0]

    assert (potential_minima_left > potential_minima and potential_minima_right > potential_minima)

def test_pseudo_potential_maxima():
    """
    Test if the unstable temperature solution is a maxima of the pseudo-potential.

    GIVEN: the default parameters
    WHEN: the pseudo_potential function is called
    THEN: the unstable temperature solution should be a maxima of the pseudo-potential
    """
    epsilon = 1
    potential_maxima_left = sr.pseudo_potential(sr.unstable_temperature_solution_default - epsilon)[0]
    potential_maxima = sr.pseudo_potential(sr.unstable_temperature_solution_default)[0]
    potential_maxima_right = sr.pseudo_potential(sr.unstable_temperature_solution_default + epsilon)[0]

    assert (potential_maxima_left < potential_maxima and potential_maxima_right < potential_maxima)

# Test for Kramer rate function

def test_kramer_rate_noise_equal_barrier():
    """
    Test if the Kramer rate function returns the correct value when the noise variance is equal to the
    barrier height.

    GIVEN: the default parameters
    WHEN: the Kramer rate function is called with noise variance equal to the barrier height
    THEN: the result should be the correct value
    """

    calculated_value = sr.Kramer_rate(noise_variance = 0.25)
    expected_value = 1/(np.sqrt(2) * np.pi)*np.exp(-1)
    assert calculated_value == pytest.approx(expected_value, rel=1e-6)

def test_kramer_rate_noise_equal_one():
    """
    Test if the Kramer rate function returns the correct value when the noise variance is equal to one.

    GIVEN: the default parameters
    WHEN: the Kramer rate function is called with noise variance equal to one
    THEN: the result should be the correct value
    """

    calculated_value = sr.Kramer_rate(noise_variance = 1)
    expected_value = 1/(np.sqrt(2) * np.pi)*np.exp(-0.25)
    assert calculated_value == pytest.approx(expected_value, rel=1e-6)

# Test for trascendal_equation_for_noise_variance function

def test_trascendental_equation_for_zero_angular_frequency():
    """
    Test if the trascendental_equation_for_noise_variance function returns the correct value when the
    forcing angular frequency is zero.

    GIVEN: the default parameters
    WHEN: the trascendental_equation_for_noise_variance function is called with forcing angular frequency
          equal to zero
    THEN: the result should be the correct value
    """

    expected_value = 4*(1/(np.sqrt(2) * np.pi)*np.exp(-0.25))**2
    calculated_value = sr.trascendental_equation_for_noise_variance(noise_variance = 1,
                                                                    forcing_angular_frequency = 0)
    assert calculated_value == pytest.approx(expected_value, rel=1e-6)

def test_trascendental_equation_for_one_angular_frequency():
    """
    Test if the trascendental_equation_for_noise_variance function returns the correct value when the
    forcing angular frequency is one.

    GIVEN: the default parameters
    WHEN: the trascendental_equation_for_noise_variance function is called with forcing angular frequency
          equal to one
    THEN: the result should be the correct value
    """

    expected_value = (4*(1/(np.sqrt(2) * np.pi)*np.exp(-0.25))**2) + 1 - 0.25 
    calculated_value = sr.trascendental_equation_for_noise_variance(noise_variance = 1,
                                                                    forcing_angular_frequency = 1) 

    assert calculated_value == pytest.approx(expected_value, rel=1e-6)   

#Test for simulate_ito function

def test_simulate_ito_time_length():
    """
    Test if the time returned by the simulate_ito function have the correct length.

    GIVEN: the default parameters with reduced number of steps (num_steps = 100)
    WHEN: the simulate_ito function is called
    THEN: the time returned should have the correct length (num_steps)
    """
    rn.seed(42)
    calculated_values = sr.simulate_ito(
        num_steps=100,
        )
    expected_value = 100
    assert len(calculated_values[0]) == expected_value

def test_simulate_ito_temperature_shape():
    """
    Test if the temperature returned by the simulate_ito function have the correct shape.

    GIVEN: the default parameters with reduced number of steps (num_steps = 100)
    WHEN: the simulate_ito function is called
    THEN: the temperature returned should have the correct shape (num_simulations, num_steps)
    """
    rn.seed(42)
    calculated_values = sr.simulate_ito(
        num_steps=100,
        )
    expected_value = (sr.num_simulations_default, 100)
    assert calculated_values[1].shape == expected_value

def test_simulate_ito_time_type():
    """
    Test if the time returned by the simulate_ito function is of type numpy.ndarray.

    GIVEN: the default parameters with reduced number of steps (num_steps = 100)
    WHEN: the simulate_ito function is called
    THEN: the time returned should be of type numpy.ndarray
    """
    rn.seed(42)
    calculated_values = sr.simulate_ito(
        num_steps=100,
        )
    assert type(calculated_values[0]) == np.ndarray

def test_simulate_ito_temperature_type():
    """
    Test if the temperature returned by the simulate_ito function is of type numpy.ndarray.

    GIVEN: the default parameters with reduced number of steps (num_steps = 100)
    WHEN: the simulate_ito function is called
    THEN: the temperature returned should be of type numpy.ndarray
    """
    rn.seed(42)
    calculated_values = sr.simulate_ito(
        num_steps=100,
        )
    assert type(calculated_values[1]) == np.ndarray


@pytest.mark.parametrize("T_start", steady_temperature_solutions)
def test_simulate_ito_no_forcing_no_noise(T_start):
    """
    Test if the temperature returned by the simulate_ito function is equal to the initial temperature
    when no forcing and no noise are applied and the initial temperature is a steady solution.

    GIVEN: the default parameters with reduced number of steps (num_steps = 100), no forcing and no noise and
              the initial temperature is a steady solution
    WHEN: the simulate_ito function is called
    THEN: the temperature returned should be equal to the initial temperature
    """

    calculated_values = sr.simulate_ito(
        T_start=T_start,
        num_steps=100,
        noise=False,
        forcing='constant'
    )
    expected_value = T_start
    assert np.all(calculated_values[1] == expected_value)

def test_simulate_ito_temperature_mean():
    """
    Test if the mean of the temperature returned by the simulate_ito function is in a valid range.

    GIVEN: the default parameters with a reduced number of simulations (num_simulations = 1) and 
            a noise variance of 0.1 (noise_variance = 0.1).
    WHEN: the simulate_ito function is called
    THEN: the mean of the temperature returned should be in a range of 10 degrees below and above the 
            first and the second temperature solutions respectively.
    """

    rn.seed(42)
    calculated_values = sr.simulate_ito(
        noise_variance = 0.1,
        num_steps = 1000,
        num_simulations = 1
    )
    assert ((np.mean(calculated_values[1]) >= sr.stable_temperature_solution_1_default-10) and
            np.mean(calculated_values[1] <= sr.stable_temperature_solution_2_default+10))


# Test binarize_temperature_inplace function

def test_binarize_one_dimentional_temperature():
    """
    Test if the binarize_temperature_inplace function returns the correct values for a one dimentional 
    array of temperatures.

    GIVEN: a one-dimensional array of temperatures and the stable temperature solutions 0 and 1
    WHEN: the binarize_temperature_inplace function is called
    THEN: the result should be an array of zeros and ones
    """

    temperature = np.array([0, 0.5, 1, 1.5, 2])
    sr.binarize_temperature_inplace(temperature = temperature,
                                    stable_temperature_solution_1=0,
                                    stable_temperature_solution_2=1)
    expected_value = np.array([0, 1, 1, 1, 1])
    assert np.array_equal(temperature, expected_value)

def test_binarize_two_dimentional_temperature():
    """
    Test if the binarize_temperature_inplace function returns the correct values for a two dimentional 
    array of temperatures.

    GIVEN: a two-dimensional array of temperatures and the stable temperature solutions 0 and 1
    WHEN: the binarize_temperature_inplace function is called
    THEN: the result should be an array of zeros and ones
    """

    temperature = np.array([[0, 0.2, 0.5, 1, 1.5, 2], [0, 0.2, 0.5, 1, 1.5, 2]])
    expected_value = np.array([[0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]])
    sr.binarize_temperature_inplace(temperature = temperature,
                                    stable_temperature_solution_1 = 0,
                                    stable_temperature_solution_2 = 1)
    assert np.array_equal(temperature, expected_value)

# Test find_peak_indices function

def test_find_peak_indices_same_peak_index():
    """
    This function tests the find_peak_indices function when the peak index is the same for 
    all simulations (rows).
    
    GIVEN: a frequency array with equal rows and a period
    WHEN: the find_peak_indices function is called
    THEN: the result should be an array of the same peak index for all simulations"""

    frequency = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]])
    period = 1.0
    calculated_values = sr.find_peak_indices(frequency, period)
    assert np.array_equal(calculated_values, np.array([0, 0]))

def test_find_peak_indices_different_peak_index():
    """
    This function tests the find_peak_indices function when the peak index is different for 
    all simulations (rows).
    
    GIVEN: a frequency array with different rows and a period
    WHEN: the find_peak_indices function is called
    THEN: the result should be an array of different peak indices for all simulations"""

    frequency = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9, 10]])
    period = 0.5
    calculated_values = sr.find_peak_indices(frequency, period)
    assert np.array_equal(calculated_values, np.array([1, 0]))

def test_find_peak_indices_approximate_peak_index():
    """
    This function tests the find_peak_indices function when the peak index is not exact.

    GIVEN: a period and a frequency array in which the value 1/period is not contained
    WHEN: the find_peak_indices function is called
    THEN: the result should be an array of approximate peak indices for all simulations
    """
    
    frequencies = np.array([[3, 4], [0, 1]])
    period = 0.5
    calculated_values = sr.find_peak_indices(frequencies, period)
    assert np.array_equal(calculated_values, np.array([0, 1]))

def test_find_peak_indices_multiple():
    """
    This function tests the find_peak_indices function when the desired frequency is 
    contained more that once.

    GIVEN: a period and a frequency array in which the value 1/period is contained more than once
    WHEN: the find_peak_indices function is called
    THEN: the result should be an array containing just the first occurrence of 1/period for each row.
    """

    frequencies = np.array([[1,1,1], [1, 1, 1]])
    period = 1
    calculated_values = sr.find_peak_indices(frequencies, period)
    assert np.array_equal(calculated_values, np.array([0, 0]))

# Test for calculate_peaks function

def test_calculate_peaks():
    """This function tests the calculate_peaks function."""

    PSD_mean = np.array([[10, 20, 30, 40, 50], [5, 15, 25, 35, 45]])
    peaks_indices = np.array([1, 3])

    expected_peaks = np.array([20, 35])
    calculated_peaks = sr.calculate_peaks(PSD_mean, peaks_indices)

    assert np.array_equal(calculated_peaks, expected_peaks)

# Test for calculate_peaks_base function

def calculate_peaks_base_normal_behaviour():
    """
    This function tests the calculate_peaks_base function when the peaks are not at the
    beginning or at the end of the frequency array and the number of neighbours is 2.

    GIVEN: a PSD_mean array, a peaks_indices array and a number of neighbours
    WHEN: the calculate_peaks_base function is called
    THEN: the result should be an array containing the mean of the values of the four neighbours 
        of the peaks (two on the right side and two on the left side).
    """
    
    PSD_mean = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                         [2, 3, 4, 5, 6, 7, 8, 9, 10]])
    peaks_indices = np.array([2, 6])

    expected_peaks_base = np.array([3, 8])

    calculated_peaks_base = sr.calculate_peaks_base(PSD_mean, peaks_indices)

    assert np.array_equal(calculated_peaks_base, expected_peaks_base)

def test_calculate_peaks_base_near_borders():
    """
    This function tests the calculate_peaks_base function when the peaks are at the beginning or 
    at the end of the frequency array and the number of neighbours is 2.

    GIVEN: a PSD_mean array, a peaks_indices array and a number of neighbours
    WHEN: the calculate_peaks_base function is called
    THEN: the result should be an array containing the mean of the values of the existing neighbours 
          of the peaks.
    """
    
    PSD_mean = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                         [2, 3, 4, 5, 6, 7, 8, 9, 10]])
    peaks_indices = np.array([0, 8])

    expected_peaks_base = np.array([2.5, 8.5])

    calculated_peaks_base = sr.calculate_peaks_base(PSD_mean, peaks_indices)

    assert np.array_equal(calculated_peaks_base, expected_peaks_base)


# Test for calculate_peak_height function

def test_calculate_peak_height_empty():
    """
    This function tests the calculate_peak_height function when the peaks and the base values are 
    empty arrays.

    GIVEN: empty arrays for peaks and base values
    WHEN: the calculate_peak_height function is called
    THEN: the result should be an empty array
    """

    peaks = np.array([])
    peaks_base = np.array([])
    peak_height = sr.calculate_peak_height(peaks, peaks_base)
    assert np.array_equal(peak_height, np.array([]))


def test_calculate_peak_height_one_element():
    """
    This function tests the calculate_peak_height function when the peaks and the base values 
    have one element.

    GIVEN: peaks and base values with one element
    WHEN: the calculate_peak_height function is called
    THEN: the result should be an array containing the difference between the two elements
    """

    peaks = np.array([5.0])
    peaks_base = np.array([2.0])
    peak_height = sr.calculate_peak_height(peaks, peaks_base)
    assert np.array_equal(peak_height, np.array([3.0]))  

def test_calculate_peak_height_multiple_elements():
    """
    This function tests the calculate_peak_height function when the peaks and the base values 
    have multiple elements.

    GIVEN: peaks and base values with multiple elements
    WHEN: the calculate_peak_height function is called
    THEN: the result should be an array containing the difference between the corresponding elements
    """
    peaks = np.array([4.0, 7.0, 10.0, 6.0])
    peaks_base = np.array([2.0, 3.0, 8.0, 4.0])
    peak_height = sr.calculate_peak_height(peaks, peaks_base)
    expected_heights = np.array([2.0, 4.0, 2.0, 2.0])
    assert np.array_equal(peak_height, expected_heights)   

def test_calculate_peak_height_zero():
    """
    This function tests the calculate_peak_height function when the peaks and the base values are all zeros

    GIVEN: peaks and base values with all zeros
    WHEN: the calculate_peak_height function is called
    THEN: the result should be an array containing zeros
    """
    peaks = np.array([0.0, 0.0, 0.0])
    peaks_base = np.array([0.0, 0.0, 0.0])
    peak_height = sr.calculate_peak_height(peaks, peaks_base)
    assert np.array_equal(peak_height, np.array([0.0, 0.0, 0.0]))

# Test for calculare_SNR function

def test_calculate_SNR_normal_behaviour():
    """
    This function tests a normal behaviour of the calculate_SNR function.

    GIVEN: peaks and base values with multiple elements
    WHEN: the calculate_SNR function is called
    THEN: the result should be an array containing the ratio between the corresponding elements multiplied by 2
    """

    peaks = np.array([4.0, 7.0, 10.0, 6.0])
    peaks_base = np.array([2.0, 3.0, 8.0, 4.0])
    expected_SNR = np.array([2*(4.0/2.0), 2*(7.0/3.0), 2*(10.0/8.0), 2*(6.0/4.0)])
    calculated_SNR = sr.calculate_SNR(peaks, peaks_base)
    assert np.array_equal(calculated_SNR, expected_SNR)

def test_calculate_SNR_zero():
    """
    This function tests the calculate_SNR function when the peaks and the base values are all zeros

    GIVEN: peaks and base values with all zeros
    WHEN: the calculate_SNR function is called
    THEN: the result should be an array containing zeros
    """

    peaks = np.array([0.0, 0.0, 0.0])
    peaks_base = np.array([0.0, 0.0, 0.0])
    calculated_SNR = sr.calculate_SNR(peaks, peaks_base)
    assert np.array_equal(calculated_SNR, np.array([0.0, 0.0, 0.0]))

def test_calculate_SNR_empty():
    """
    This function tests the calculate_SNR function when the peaks and the base values are empty arrays

    GIVEN: empty arrays for peaks and base values
    WHEN: the calculate_SNR function is called
    THEN: the result should be an empty array
    """

    peaks = np.array([])
    peaks_base = np.array([])
    calculated_SNR = sr.calculate_SNR(peaks, peaks_base)
    assert np.array_equal(calculated_SNR, np.array([]))