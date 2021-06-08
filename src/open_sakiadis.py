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
    a = solution_utils.get_series(conditions)
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
    def theta(t):
        if t == 0:
            return conditions.theta0
        return sum(A*np.exp(-n*t/np.sqrt(2)) for (n, A) in enumerate(coeffs))
    return theta
    