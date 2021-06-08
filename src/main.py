from matplotlib import pyplot as plt
import numpy as np
import solution_utils
import math

import old_approximant
import critical_additive
import critical_multiplicative
import exp_first_order
import writeup_sinh_cos
import writeup_sinh_poly
import pure_sakiadis
import open_sakiadis
import trig_fo_sak
import poly_fo_sak
import trig_to_sak
import poly_to_sak
import trig_so_sak
import poly_so_sak
import exp_fo_mult
import near_recip_oc_sak
import nr_oc_fo_poly_sak
import nr_oc_fo_trig_sak
import nr_oc_so_poly_sak
import nr_oc_so_trig_sak
import nr_oc_to_poly_sak
import nr_oc_to_trig_sak
import tanh_fo_mult
import tanh_oc_fo_sak
import tanh_nr_oc_fo_sak
import tanh_oc_so_sak
import tanh_nr_oc_so_sak
import tanh_oc_to_sak
import tanh_nr_oc_to_sak

APPROX_COLORS = ['r', 'b', 'm', 'y', 'c']
ALL_APPROXIMANTS = [old_approximant.get_approximant, 
                    critical_additive.get_approximant,
                    critical_multiplicative.get_approximant,
                    exp_first_order.get_approximant,
                    writeup_sinh_cos.get_approximant,
                    writeup_sinh_poly.get_approximant,
                    pure_sakiadis.get_approximant,
                    open_sakiadis.get_approximant,
                    trig_fo_sak.get_approximant,
                    poly_fo_sak.get_approximant,
                    trig_to_sak.get_approximant,
                    poly_to_sak.get_approximant,
                    trig_so_sak.get_approximant,
                    poly_so_sak.get_approximant, 
                    exp_fo_mult.get_approximant, 
                    near_recip_oc_sak.get_approximant,
                    nr_oc_fo_poly_sak.get_approximant,
                    nr_oc_fo_trig_sak.get_approximant,
                    nr_oc_so_poly_sak.get_approximant,
                    nr_oc_so_trig_sak.get_approximant,
                    nr_oc_to_poly_sak.get_approximant,
                    nr_oc_to_trig_sak.get_approximant,
                    tanh_fo_mult.get_approximant, 
                    tanh_oc_fo_sak.get_approximant, 
                    tanh_nr_oc_fo_sak.get_approximant,
                    tanh_oc_so_sak.get_approximant,
                    tanh_nr_oc_so_sak.get_approximant,
                    tanh_oc_to_sak.get_approximant,
                    tanh_nr_oc_to_sak.get_approximant]

APPROXIMANTS = [ALL_APPROXIMANTS[-1]]

def numerical_plot(conditions):
    '''
    Plots the numerical solution on the plot
    '''
    times, angles = solution_utils.numerical_solve(conditions)
    plot_times = times[::conditions.plotresrat]
    plot_angles = angles[::conditions.plotresrat]
    plt.plot(plot_times, plot_angles, 'k.')
    

def series_plot(conditions):
    '''
    Plots the power series solution on the plot
    '''
    #get the raw points. Note that this functional approach can be spread up by
    #broadcasting, but we retain it for aid of use at individual points, which
    #is an advantage of the method of asymptotic approximants
    series_sol = solution_utils.get_series_function(conditions)
    times = np.linspace(0, conditions.TT, math.ceil(conditions.TT/conditions.dt))
    thetas = np.array([series_sol(t) for t in times])
    #make sure it doesn't go too far
    good_indices = list(filter(lambda x: -.05 < thetas[x] < 1.1 * np.pi, 
                          range(len(times))))
    times = [times[gi] for gi in good_indices]
    thetas = [thetas[gi] for gi in good_indices]
    plt.plot(times, thetas, 'g')
    
def plot_approximants(approxes, conditions):
    for (ind, approx) in enumerate(approxes):
        approximant = approx(conditions)
        times = np.linspace(0, conditions.TT, math.ceil(conditions.TT/conditions.dt))
        thetas = [solution_utils.clamp_abs(approximant(t), 1.2*np.pi) for t in times]
        colorstr = APPROX_COLORS[ind % len(APPROX_COLORS)]
        plt.plot(times, thetas, colorstr)
    
def plot_start(conditions):
    '''
    Plots all of the numerical solution, series solution, and apporximant 
    solution for the given conditions

    '''
    numerical_plot(conditions)
    series_plot(conditions)
 
if __name__ == "__main__":
    m_conditions = solution_utils.Conditions(N = 10)
    plot_start(m_conditions)
    plot_approximants(APPROXIMANTS, m_conditions)