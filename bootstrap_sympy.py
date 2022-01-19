# About bootstrap_sympy:
# - sympy_solve_intervals : use sympy.solvers.inequalities to solve
# - plot_energy_interval  : use plt to plot energy intervals

import os, time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sympy as sp
from sympy import Poly, Rational
from sympy.sets.sets import Intersection, Union
from sympy.solvers.inequalities import solve_poly_inequality, solve_rational_inequalities

def sympy_solve_intervals(matrix, config, mode='Rational', from_checkpoint=True):
    '''
        matrix -> class : the potential we want to solve
        config -> dict : contains information computing
        hyperparameters -> dict : contains information of hyperparameters
        from_checkpoint -> bool : whether to start from saving point
    '''
    round = config['round'] # maximum size of determinant we want to compute
    energy_threshold = config['threshold'] # sympy can't solve too small intervals, so we keep small intervals
    initial_interval = config['initial_interval'] # initial interval to be start with
    
    if os.path.exists(config['npy_energy_intervals']):
        # check if data already exists
        energy_intervals = list(np.load(config['npy_energy_intervals'], allow_pickle=True))
        survived_interval = energy_intervals[-1]
        print(f"Checkpoint exists, checkpoint max K={len(energy_intervals)}\n")
    else:
        energy_intervals = [] # record the energy interval in each round (each LxL determinant)
        survived_interval = initial_interval # initial energy interval, set to be positive
    if len(energy_intervals) < round:
        for i in range(1+len(energy_intervals), round+1):
            print(f"Now calculating determinant size {i}x{i}")

            # solve inequalities
            t_start = time.time()
            det = matrix.submatrix(i).det()
            t_det = time.time()
            print(f"Calculating determinant time : {t_det-t_start : .2f}")
            if mode == 'Poly':
                possible_intervals = solve_poly_inequality(Poly(det, matrix.E), '>=')
            elif mode == 'Rational':
                det = sp.simplify(det)
                numerator, denominator = det.as_numer_denom()
                numerator, denominator = sp.Poly(numerator, matrix.E), sp.Poly(denominator, matrix.E)
                possible_intervals = solve_rational_inequalities([[((numerator, denominator), '>=')]])
            else:
                assert False, f"Invalid mode : {mode}"
            t_ieq = time.time()
            print(f"Solving  inequalities time : {t_ieq - t_det:.2f}")

            # intersect the new intervals with old intervals
            positive_intervals = []
            if type(possible_intervals) == Union:
                # solve_rational_inequalities returns Union
                for interval in possible_intervals.args:
                    positive_intervals.append(Intersection(survived_interval, interval))
            elif type(possible_intervals) == list:
                # solve_poly_inequality returns list
                for interval in possible_intervals:
                    positive_intervals.append(Intersection(survived_interval, interval))
            else:
                positive_intervals.append(Intersection(survived_interval, possible_intervals))
            survived_interval = Union(*positive_intervals)
            print(f"Survived interval = {sp.N(survived_interval)}\n")

            energy_intervals.append(sp.N(survived_interval))
    
    return energy_intervals

def plot_energy_interval(energy_intervals, energy_eigenvalues, x_ticks, config):
    '''
        energy_intervals -> list : intervals solved with sympy inequalities
        energy_eigenvalues -> list : analytical energy eigenvalues
        x_ticks -> list : ticks at x-axis
        config -> dict : contains information of hyperparameters
    '''

    x_inf = config['x_inf'] # infimum of x plot
    x_sup = config['x_sup'] # supreme of x plot
    plot_step = config['plot_step'] # how often we want to plot the energy interval

    fig, ax = plt.subplots(figsize=(12,8))
    plt.xticks(energy_eigenvalues, x_ticks)
    ax.set_xlim(x_inf, x_sup)

    # plot verticle line of energy eigenvalues
    for En in energy_eigenvalues:
        plt.axvline(x=En, color='white')

    color_list = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i in range(0, len(energy_intervals), plot_step):
        y_plot = [f"$K$={i+1}", f"$K$={i+1}"]
        if type(energy_intervals[i]) == sp.Interval:
            # only one interval
            x_plot = [float(energy_intervals[i].inf), min(1e9, float(energy_intervals[i].sup))]
            ax.plot(x_plot, y_plot, marker='o', color = 'tab:'+color_list[(i//plot_step)%len(color_list)])
        else:
            # union of intervals
            for plot_interval in energy_intervals[i].args:
                x_plot = [float(plot_interval.inf), min(1e9, float(plot_interval.sup))]
                ax.plot(x_plot, y_plot, marker='o', color = 'tab:'+color_list[(i//plot_step)%len(color_list)])
    return fig, ax