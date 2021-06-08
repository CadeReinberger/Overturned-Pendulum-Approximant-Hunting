import numpy as np
import solution_utils
import taylor_lib #didn't need it here because pre-optimized. 
import copy

def get_critical_series(conditions):
    '''
    returns the critical series approprately
    '''
    new_conds = copy.deepcopy(conditions)
    new_conds.omega0 = np.sqrt(2)
    return solution_utils.get_series(new_conds)

def get_approx_coeffs(conditions):
    '''
    Returns the coefficients of the approximant. 
    '''
    assert(conditions.theta0==0)
    assert(np.isclose(conditions.g/conditions.l, .5))
    a = solution_utils.get_series(conditions)
    cs = get_critical_series(conditions)
    a_hat = a - cs
    return a_hat

    
def get_approximant(conditions):
    '''
    Returns a function that computes the approximant for a given time
    '''
    assert(conditions.theta0==0)
    assert(np.isclose(conditions.g/conditions.l, .5))
    coeffs = get_approx_coeffs(conditions)
    def theta(t):
        if t == 0:
            return conditions.theta0
        return  2*np.arcsin(np.tanh(t/np.sqrt(2))) + sum(a * (t ** i) for (i, a) in enumerate(coeffs))
    return theta
    