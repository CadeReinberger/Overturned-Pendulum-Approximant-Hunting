import numpy as np
import solution_utils
import taylor_lib 

def get_approx_coeffs(conditions):
    '''
    Returns the coefficients of the approximant. 
    '''
    assert(conditions.theta0==0)
    assert(np.isclose(conditions.g/conditions.l, .5))
    #get the half-period and speed at top
    T_2 = solution_utils.get_half_period(conditions)
    theta_dot_tau = solution_utils.get_omega_tau(conditions)
    #get the series solution
    a = solution_utils.get_series(conditions)
    #subtract away the terms out front, by expanding sinh into exps. 
    plus_exp = taylor_lib.scale(taylor_lib.exp(conditions.N), 1/np.sqrt(2))
    min_exp = taylor_lib.scale(taylor_lib.exp(conditions.N), -1/np.sqrt(2))
    plus_coeff = .5*np.exp(-T_2/np.sqrt(2))
    min_coeff = .5*np.exp(T_2/np.sqrt(2))
    addend = np.sqrt(2) * theta_dot_tau * (plus_coeff * plus_exp - min_coeff * min_exp)
    addend[0] += np.pi
    #divide by the coefficient
    cosine = taylor_lib.scale(taylor_lib.cos(conditions.N), .5*np.pi/T_2)
    mult_recip = taylor_lib.series_power(cosine, -3)
    #mult_recip = taylor_lib.series_recip(taylor_lib.series_multiply(taylor_lib.series_multiply(cosine, cosine), cosine))
    a_hat = taylor_lib.series_multiply(a - addend, mult_recip)
    #return the result
    return a_hat

    
def get_approximant(conditions):
    '''
    Returns a function that computes the approximant for a given time
    '''
    assert(conditions.theta0==0)
    assert(np.isclose(conditions.g/conditions.l, .5))
    coeffs = get_approx_coeffs(conditions)
    #get the half-period and speed at top
    T_2 = solution_utils.get_half_period(conditions)
    theta_dot_tau = solution_utils.get_omega_tau(conditions)
    def theta(t):
        if t == 0:
            return conditions.theta0
        return np.pi  + theta_dot_tau*np.sqrt(2)*np.sinh((t-T_2)/np.sqrt(2)) + (np.cos(.5*np.pi*t/T_2)**3)*sum(a * (t ** i) for (i, a) in enumerate(coeffs))
    return theta
    