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

def sympy_solve_intervals(matrix, config, mode='Rational', keep=False, from_checkpoint=True):
    '''
        matrix -> class : the potential we want to solve
        config -> dict : contains information computing
        hyperparameters -> dict : contains information of hyperparameters
        keep -> bool : whether to keep the small interval
        from_checkpoint -> bool : whether to start from saving point
    '''
    round = config['round'] # maximum size of determinant we want to compute
    energy_threshold = config['threshold'] # sympy can't solve too small intervals, so we keep small intervals
    initial_interval = config['initial_interval'] # initial interval to be start with
    
    if os.path.exists(config['npy_energy_intervals']) and os.path.exists(config['npy_confirmed_intervals']):
        # check if data already exists
        energy_intervals = list(np.load(config['npy_energy_intervals'], allow_pickle=True))
        confirmed_intervals = np.load(config['npy_confirmed_intervals'], allow_pickle=True)
        if confirmed_intervals != None:
            confirmed_intervals = confirmed_intervals.item()
        survived_interval = energy_intervals[-1]
        print(f"Checkpoint exists, checkpoint max K={len(energy_intervals)}\n")
    else:
        energy_intervals = [] # record the energy interval in each round (each LxL determinant)
        confirmed_intervals = None # small intervals that we confirm to be the energy eigenvalues
        survived_interval = initial_interval # initial energy interval, set to be positive
    if len(energy_intervals) < round:
        for i in range(1+len(energy_intervals), round+1):
            print(f"Now calculating determinant size {i}x{i}")

            # solve inequalities
            t_start = time.time()
            det = matrix.submatrix(i).det()
            if mode == 'Poly':
                possible_intervals = solve_poly_inequality(Poly(det, matrix.E), '>=')
            elif mode == 'Rational':
                det = sp.simplify(det)
                print("Finish det")
                numerator, denominator = det.as_numer_denom()
                print("Finish extracting numerator and denominator ...")
                numerator, denominator = sp.Poly(numerator, matrix.E), sp.Poly(denominator, matrix.E)
                possible_intervals = solve_rational_inequalities([[((numerator, denominator), '>=')]])
            else:
                assert False, f"Invalid mode : {mode}"
            t_end = time.time()
            print(f"Time cost = {t_end - t_start:.2f}")

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

            # check intervals that can't be calculated by sympy due to too small intervals
            for interval in survived_interval.args:
                if (type(interval)==sp.Interval) and (interval.sup - interval.inf < energy_threshold):
                    if confirmed_intervals ==  None:
                        confirmed_intervals = interval
                    else:
                        confirmed_intervals =  Union(confirmed_intervals, interval)
            if confirmed_intervals != None and keep:
                survived_interval = Union(survived_interval, confirmed_intervals)

            energy_intervals.append(sp.N(survived_interval))
    
    return energy_intervals, confirmed_intervals

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
            ax.plot(x_plot, y_plot, marker='o', color = 'tab:'+color_list[(i)%len(color_list)])
        else:
            # union of intervals
            for plot_interval in energy_intervals[i].args:
                x_plot = [float(plot_interval.inf), min(1e9, float(plot_interval.sup))]
                ax.plot(x_plot, y_plot, marker='o', color = 'tab:'+color_list[(i)%len(color_list)])
    return fig, ax