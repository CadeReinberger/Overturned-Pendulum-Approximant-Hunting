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
    T_2 = solution_utils.get_half_period(conditions)
    #subtract the leading term
    a[0] -= np.pi
    #get the coefficient
    exp_term = np.exp(-T_2*np.sqrt(2)) * taylor_lib.scale(taylor_lib.exp(conditions.N), np.sqrt(2))
    one = taylor_lib.fill(1, conditions.N)
    tanh = taylor_lib.series_multiply(exp_term - one, taylor_lib.series_recip(exp_term + one))
    #divide by the coefficient
    a_hat = taylor_lib.series_multiply(a, taylor_lib.series_recip(tanh))
    #return the result
    return a_hat

    
def get_approximant(conditions):
    '''
    Returns a function that computes the approximant for a given time
    '''
    assert(conditions.theta0==0)
    assert(np.isclose(conditions.g/conditions.l, .5))
    coeffs = get_approx_coeffs(conditions)
    T_2 = solution_utils.get_half_period(conditions)
    def theta(t):
        if t == 0:
            return conditions.theta0
        return np.pi + np.tanh((t-T_2)/np.sqrt(2))*sum(a * (t ** i) for (i, a) in enumerate(coeffs))
    return theta
    