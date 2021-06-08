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
    #subtract the term out front.
    a[0] -= np.pi
    #divide by the coefficient
    cosine = taylor_lib.scale(taylor_lib.cos(conditions.N), .5*np.pi/T_2)
    a = taylor_lib.series_multiply(a, taylor_lib.series_recip(cosine))
    #solve the vandermonde system
    a_hat = np.array([a[j]*factorial(j)*((-np.sqrt(2))**j) for j in range(conditions.N)])
    A = solution_utils.VDSolve(np.array(list(range(1, conditions.N+1))), a_hat)
    return A

    
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
        return np.pi + np.cos(.5*np.pi*t/T_2)*sum(A*np.exp(-(n+1)*t/np.sqrt(2)) for (n, A) in enumerate(coeffs))
    return theta
    