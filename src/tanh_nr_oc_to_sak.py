import numpy as np
import solution_utils
import taylor_lib
from scipy.special import factorial

def get_approx_coeffs(conditions):
    '''
    Returns the coefficients of the approximant. 
    '''
    assert(conditions.theta0==0)
    assert(np.isclose(conditions.g/conditions.l, .5))
    #get the series solution
    a = solution_utils.get_series(conditions)
    T_2 = solution_utils.get_half_period(conditions)
    wt = solution_utils.get_omega_tau(conditions)
    #get the tanh
    exp_term = np.exp(-T_2*np.sqrt(2)) * taylor_lib.scale(taylor_lib.exp(conditions.N), np.sqrt(2))
    one = taylor_lib.fill(1, conditions.N)
    tanh = taylor_lib.series_multiply(exp_term - one, taylor_lib.series_recip(exp_term + one))
    #subtract the addend
    addend = (wt * np.sqrt(2)) * tanh
    addend[0] += np.pi
    a = a - addend
    #divide appropriately
    a = taylor_lib.series_recip(a)
    a = taylor_lib.series_multiply(a, taylor_lib.series_power(tanh, 3))
    #solve the Vandermonde system.
    a_hat = np.array([a[j]*factorial(j)*((-np.sqrt(2))**j) for j in range(conditions.N+1)])
    A = solution_utils.VDSolve(np.array(list(range(conditions.N+1))), a_hat)
    return A

    
def get_approximant(conditions):
    '''
    Returns a function that computes the approximant for a given time
    '''
    assert(conditions.theta0==0)
    assert(np.isclose(conditions.g/conditions.l, .5))
    coeffs = get_approx_coeffs(conditions)
    T_2 = solution_utils.get_half_period(conditions)
    wt = solution_utils.get_omega_tau(conditions)
    def theta(t):
        if t == 0:
            return conditions.theta0
        return np.pi + (wt*np.sqrt(2))*np.tanh((t-T_2)/np.sqrt(2)) + (np.tanh((t-T_2)/np.sqrt(2))**3)/sum(A*np.exp(-n*t/np.sqrt(2)) for (n, A) in enumerate(coeffs))
    return theta
    