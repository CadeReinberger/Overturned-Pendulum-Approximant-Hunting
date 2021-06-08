import numpy as np
import solution_utils
import taylor_lib

def get_approx_coeffs(conditions):
    '''
    Returns the coefficients of the approximant. 
    '''
    assert(conditions.theta0==0)
    assert(np.isclose(conditions.g/conditions.l, .5))
    #get the series solution
    a = solution_utils.get_series(conditions)
    #subtract away the terms out front
    addend = -4 * taylor_lib.scale(taylor_lib.exp(conditions.N), -1/np.sqrt(2))
    addend[0] += np.pi
    #divide by the coefficient
    mult = taylor_lib.scale(taylor_lib.exp(conditions.N), -np.sqrt(2))
    a_hat = taylor_lib.series_multiply(a - addend, taylor_lib.series_recip(mult))
    #return the result
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
        return np.pi - 4*np.exp(-t/np.sqrt(2)) + np.exp(-t*np.sqrt(2))*sum(a * (t ** i) for (i, a) in enumerate(coeffs))
    return theta
    