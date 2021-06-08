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
    cosine = taylor_lib.scale(taylor_lib.cos(conditions.N), .5*np.pi/T_2)
    sine = taylor_lib.scale(taylor_lib.sin(conditions.N), .5*np.pi/T_2)
    sine_sq = taylor_lib.series_multiply(sine, sine)
    addend = -(2*T_2*wt/np.pi) * taylor_lib.series_multiply(cosine, sine_sq)
    addend[0] += np.pi 
    a = a - addend
    #divide by the coefficient    
    mult_recip = taylor_lib.series_power(cosine, -3)
    a = taylor_lib.series_multiply(a, mult_recip)
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
    wt = solution_utils.get_omega_tau(conditions)
    def theta(t):
        if t == 0:
            return conditions.theta0
        return np.pi - (2*T_2*wt/np.pi)*np.cos(.5*np.pi*t/T_2)*(np.sin(.5*np.pi*t/T_2)**2) + (np.cos(.5*np.pi*t/T_2)**3)*sum(A*np.exp(-(n+1)*t/np.sqrt(2)) for (n, A) in enumerate(coeffs))
    return theta
    