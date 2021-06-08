import numpy as np
import solution_utils
import taylor_lib #didn't need it here because pre-optimized. 

def get_approx_coeffs(conditions):
    '''
    Returns the coefficients of the approximant. 
    '''
    assert(conditions.theta0==0)
    T_2 = solution_utils.get_half_period(conditions)
    #get the series solution
    a = solution_utils.get_series(conditions)
    b = np.copy(a)
    #modify the early terms to subtract jet at T/2
    theta_dot_tau = solution_utils.get_omega_tau(conditions)
    b[0] = a[0] - np.pi + T_2 * theta_dot_tau
    b[1] = a[1] - theta_dot_tau
    #next, we actually compute the caychy's product rule
    a_hat = np.zeros(conditions.N+1)
    for n in range(len(a_hat)):
        a_hat[n] = sum(b[n-k] * (k+1) * (T_2 ** (-k-2)) for k in range(n+1))
    return a_hat

    
def get_approximant(conditions):
    '''
    Returns a function that computes the approximant for a given time
    '''
    assert(conditions.theta0==0)
    T_2 = solution_utils.get_half_period(conditions)
    coeffs = get_approx_coeffs(conditions)
    theta_dot_tau = np.sqrt(conditions.omega0**2 - 4*conditions.g/conditions.l)
    def theta(t):
        if t == 0:
            return conditions.theta0
        return np.pi + (t - T_2) * theta_dot_tau + (T_2 - t)**2 * sum(a * (t ** i) for (i, a) in enumerate(coeffs))
    return theta
    