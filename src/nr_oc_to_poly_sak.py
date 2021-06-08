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
    #get the series solution and half-period
    a = solution_utils.get_series(conditions)
    T_2 = solution_utils.get_half_period(conditions)
    wt = solution_utils.get_omega_tau(conditions)
    #subtract the term out front.
    lin_term = taylor_lib.fill(-T_2, 1, conditions.N)
    addend = wt * lin_term
    addend[0] += np.pi  
    a = a - addend
    #multiply appropriately
    a = taylor_lib.series_recip(a)
    a = taylor_lib.series_multiply(a, taylor_lib.series_power(lin_term, 3))
    #Solve for the Sakiadis coefficients    
    a_hat = np.array([a[j]*factorial(j)*((-np.sqrt(2))**j) for j in range(conditions.N+1)])
    A = solution_utils.VDSolve(np.array(list(range(conditions.N+1))), a_hat)
    return A

    
def get_approximant(conditions):
    '''
    Returns a function that computes the approximant for a given time
    '''
    assert(conditions.theta0==0)
    assert(np.isclose(conditions.g/conditions.l, .5))
    T_2 = solution_utils.get_half_period(conditions)
    wt = solution_utils.get_omega_tau(conditions)
    coeffs = get_approx_coeffs(conditions)
    def theta(t):
        if t == 0:
            return conditions.theta0
        return np.pi + wt*(t-T_2) + ((t-T_2)**3)/(sum(A*np.exp(-n*t/np.sqrt(2)) for (n, A) in enumerate(coeffs)))
    return theta
    