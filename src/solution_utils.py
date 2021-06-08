from dataclasses import dataclass 
import numpy as np
from scipy.special import ellipk
from scipy.integrate import odeint

'''
Code to compute the trajectory of a simple pendulum (starting from the bottom 
and overturning) using the method of asymptotic approximants. 
'''

#This, as a global variable, gets the method with which the period is computed
CEI_FUNCTION = ellipk

def _half_period(wo, g, l):
    '''
    A silly helper to ignore. 
    '''
    T_2 = (2/wo) * CEI_FUNCTION(((4*g)/(l * wo**2)))
    return T_2

def get_half_period(conditions):
    '''
    Returns the period of the associated problem. Just a wrapper to scale
    appropriately
    '''
    wo = conditions.omega0
    assert(conditions.theta0 == 0)
    T_2 = _half_period(wo, conditions.g, conditions.l)
    return T_2

@dataclass
class Conditions:
    '''
    A dataclass to house the conditions of a pendulum problem
    '''
    theta0: float = 0 #intial angle. 
    omega0: float = 1.45 #1.45 #2**(3/4) + .01 #intial angular velocity. Must be 0 for some functions. 
    l: float = 20 #length of the pendulum string
    g: float = 10 #gravitational field
    TT: float = _half_period(1.45, 10, 20) #how far out in time to plot the results to
    dt: float = .0001 #timescale used for Euler's method numerical solution
    plotresrat: int = 200 #how often numerical points are included in plot
    N: int = 5 #what order to compute the series and approximant to. 

def numerical_solve(conditions):
    '''
    Numerical solution using adaptive Runge-Kutta
    '''
    times = np.arange(0, conditions.TT, conditions.dt)
    kappa = conditions.g / conditions.l
    deriv = lambda x, t : (x[1], -kappa * np.sin(x[0]))
    init_cond = (conditions.theta0, conditions.omega0)
    ps = odeint(deriv, init_cond, times)
    thetas = ps[:,0]
    return times, thetas

def get_omega_tau(conditions):
    return np.sqrt(conditions.omega0**2 - 4*conditions.g/conditions.l)

def get_series(conditions):
    '''
    Returns the coefficients of the Taylors series solution to the pendulum 
    problem
    '''
    #these are in equations (#). Initialize the solution, and its sine and 
    #cosine to be filled with zeros. 
    a = np.zeros(conditions.N + 1)
    C = np.zeros(conditions.N + 1) 
    S = np.zeros(conditions.N + 1)
    #set the initial conditions 
    a[0] = conditions.theta0
    a[1] = conditions.omega0;
    kappa=conditions.g/conditions.l;
    C[0] = np.cos(a[0]) 
    S[0] = np.sin(a[0]);
    #now apply the key equation (#)
    for n in range(conditions.N - 1):
        C[n+1] = -sum((k+1)*a[k+1]*S[n-k] for k in range(n+1))/(n+1)
        S[n+1] = sum((k+1)*a[k+1]*C[n-k] for k in range(n+1))/(n+1)
        a[n+2] = -kappa*S[n]/((n+1)*(n+2)) 
    #and the a is our result
    return a    

def get_series_function(conditions):
    '''
    Returns a function that will compute the series solution value of theta
    for a given time t
    '''
    coeffs = get_series(conditions)
    def theta(t):
        if t == 0:
            return coeffs[0]
        return sum(a*(t**i) for (i, a) in enumerate(coeffs))
    return theta

def VDSolve(alpha, b):
    '''
    uses the bjork-pereyra algorithm for solving vandermonde systems. All the
    previous papers have used an explicit inversion formula, but I'm worried 
    about the ill-conditioning of the numerical methods, so it's this for now. 
    source:
        https://github.com/nschloe/vandermonde/blob/master/vandermonde/main.py
    '''
    if isinstance(alpha, int):
        alpha = np.array(range(1, alpha + 1))
    n = len(b)
    x = b.copy()
    for k in range(n):
        x[k + 1 : n] -= alpha[k] * x[k : n - 1]
    for k in range(n - 1, 0, -1):
        x[k:n] /= alpha[k:n] - alpha[: n - k]
        x[k - 1 : n - 1] -= x[k:n]
    return x

def clamp_abs(num, clamp):
    '''
    clamps num to have at most the absolute value of clamp
    '''
    clamp_raw = lambda x, c : min(max(-c, x), c)
    return clamp_raw(num, np.abs(clamp))