# About bootstrap_sympy:
# - sympy_solve_intervals : use sympy.solvers.inequalities to solve
# - plot_energy_interval  : use plt to plot energy intervals

import os, time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sympy as sp
from sympy import Poly
from sympy.sets.sets import Intersection, Union
from sympy.solvers.inequalities import solve_poly_inequality

def sympy_solve_intervals(matrix, config, det_indep=False):
    '''
        matrix -> class   : the potential we want to solve
        config -> dict    : contains information of hyperparameters
        det_indep -> bool : whether intersect new intervals with old intervals
    '''
    k = config['k'] # constant coefficient of the potential
    round = config['round'] # maximum size of determinant we want to compute
    plot_step = config['plot_step'] # how often we want to plot the energy interval
    energy_threshold = config['threshold'] # sympy can't solve too small intervals, so we keep small intervals
    initial_interval = config['initial_interval'] # initial interval to be start with
    
    energy_intervals = [] # record the energy interval in each round (each LxL determinant)
    confirmed_intervals = None # small intervals that we confirm to be the energy eigenvalues
    round_interval = initial_interval # initial energy interval, set to be positive
    for i in range(1, round+1):
        print(f"Now calculating determinant size {i}x{i}")

        # solve inequalities
        t_start = time.time()
        possible_intervals = solve_poly_inequality(Poly(matrix.submatrix(i).det().evalf(subs={matrix.k:k}), matrix.E), '>=')
        t_end = time.time()
        print(f"Time cost = {t_end - t_start:.2f}")

        # intersect the new intervals with old intervals
        positive_intervals = []
        for interval in possible_intervals:
            if det_indep:
                positive_intervals.append(Intersection(initial_interval, interval))
            else:
                positive_intervals.append(Intersection(round_interval, interval))
        round_interval = Union(*positive_intervals)
        print(f"Round interval = {sp.N(round_interval)}\n")

        # check intervals that can't be calculated by sympy due to too small intervals
        for interval in round_interval.args:
            if (type(interval)==sp.Interval) and (interval.sup - interval.inf < energy_threshold):
                if confirmed_intervals ==  None:
                    confirmed_intervals = interval
                else:
                    confirmed_intervals =  Union(confirmed_intervals, interval)
        if confirmed_intervals != None:
            round_interval = Union(round_interval, confirmed_intervals)

        # record intervals to be plotted
        if i%plot_step == 0:
            energy_intervals.append(sp.N(round_interval))
    
    return energy_intervals, sp.N(confirmed_intervals)

def plot_energy_interval(energy_intervals, energy_eigenvalues, config):
    '''
        energy_intervals -> list : intervals solved with sympy inequalities
        energy_eigenvalues -> list : analytical energy eigenvalues
        config -> dict : contains information of hyperparameters
    '''

    k = config['k'] # constant coefficient of the potential
    x_inf = config['x_inf'] # infimum of x plot
    x_sup = config['x_sup'] # supreme of x plot
    plot_step = config['plot_step'] # how often we want to plot the energy interval

    fig, ax = plt.subplots()
    ax.set_xlim(x_inf, x_sup)

    # plot verticle line of energy eigenvalues
    for En in energy_eigenvalues:
        plt.axvline(x=En, color='white')

    color_list = ['b', 'g', 'r', 'c', 'm', 'y']
    for i in range(len(energy_intervals)):
        y_plot = ['L='+str(plot_step*(i+1)), 'L='+str(plot_step*(i+1))]
        if type(energy_intervals[i]) == sp.Interval:
            # only one interval
            x_plot = [float(energy_intervals[i].inf), min(1e9, float(energy_intervals[i].sup))]
            ax.plot(x_plot, y_plot, marker='o', color = color_list[i])
        else:
            # union of intervals
            for plot_interval in energy_intervals[i].args:
                x_plot = [float(plot_interval.inf), min(1e9, float(plot_interval.sup))]
                ax.plot(x_plot, y_plot, marker='o', color = color_list[i])
    return fig, ax