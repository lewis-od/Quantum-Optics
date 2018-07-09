import numpy as np
from copy import copy
from .. import states

def min_truncation_coefficients(state, epsilon):
    """
    Returns the minimum truncation value such that the norm of the difference
    between the analytic and operator versions of the state is less than epsilon
    :param state: The state to test
    :param epsilon: The error threshold
    """
    if state.type == 'fock':
        return state.n
    test_state = copy(state)
    test_state.T = 1
    error = 1
    while error > epsilon:
        test_state.T += 1
        test_state.analytic = True
        analytic = test_state.data
        test_state.analytic = False
        matrix = test_state.data
        error = np.linalg.norm(analytic - matrix)
    return test_state.T

def min_truncation_norm(state, epsilon):
    """
    Returns the minimum truncation value required for the state to be normalised
    to 1+/-epsilon
    :param state: The state to test
    :param epsilon: The error threshold
    """
    if state.type == 'fock':
        return state.n
    test_state = copy(state)
    test_state.T = 1
    error = 1
    while error > epsilon:
        test_state.T += 1
        error = np.abs(test_state.norm() - 1)
    return test_state.T

def min_truncation_photon_number(state, epsilon):
    """
    Returns the minimum truncation required for the difference between the
    average photon number calculated analytically and using the truncated
    number operator to be less than epsilon
    :param state: The state to test
    :param epsilon: The error threshold
    """
    test_state = copy(state)
    actual = None
    if test_state.type == 'fock':
        return test_state.n
    elif test_state.type == 'coherent':
        actual = np.abs(test_state.alpha) ** 2
    elif test_state.type == 'squeezed':
        actual = np.sinh(np.abs(test_state.z))**2
    if actual is None:
        raise ValueError("Invalid state type supplied.")
    test_state.T = 1
    error = 1
    while error > epsilon:
        test_state.T += 1
        error = np.abs(test_state.avg_n() - actual)
    return test_state.T
