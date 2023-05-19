########################################################################################
#                                                                                      #
#  Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).  #
#  Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains    #
#  certain rights in this software.                                                    #
#                                                                                      #
#  This software is released under the license detailed in the file, LICENSE.          #
#                                                                                      #
########################################################################################

import numpy as np
import time
from numba import jit
from spitfire import PIController, odesolve


@jit(nopython=True)
def tridiag(a, b, c, d, x, cp, dp, n):
    '''Solve a tridiagonal system

    Args:
        a (numpy array, length n): subdiagonal
        b (numpy array, length n): main diagonal
        c (numpy array, length n): superdiagonal
        d (numpy array, length n): rhs
        x (numpy array, length n): the answer
        cp (numpy array, length n): temp array
        dp (numpy array, length n): temp array
        n (int): number of equations

    Returns:
        x (numpy array, length n): the answer
    '''

    # Initialize cp and dp
    cp[0] = c[0]/b[0]
    dp[0] = d[0]/b[0]

    # Solve for vectors cp and dp
    for i in range(1,n):
        m = b[i]-cp[i-1]*a[i]
        cp[i] = c[i]/m
        dp[i] = (d[i]-dp[i-1]*a[i])/m

    # Initialize x
    x[n-1] = dp[n-1]

    # Solve for x from the vectors cp and dp
    for i in range(n-2, -1, -1):
        x[i] = dp[i]-cp[i]*x[i+1]

    return x


def steady_solve(eqn_sys):
    eqn_sys.steady_solve()
    print('Conduction Solve Time: {:0.2f} s'.format(eqn_sys.time_conduction))


def transient_solve(eqn_sys, verbose=True):
    '''Transient solve using Spitfire
    '''
    if eqn_sys.fixed_step:
        step_size = eqn_sys.dt
    else:
        step_size = PIController(target_error=eqn_sys.target_error)

    t_st = time.time()
    q = odesolve(eqn_sys.right_hand_side,
                 eqn_sys.initial_state,
                 output_times=eqn_sys.t,
                 linear_setup=eqn_sys.setup_superlu,
                 linear_solve=eqn_sys.solve_superlu,
                 norm_weighting=eqn_sys.norm_weighting,
                 step_size=step_size,
                 linear_setup_rate=eqn_sys.linear_setup_rate,
                 verbose=verbose,
                 log_rate=100,
                 show_solver_stats_in_situ=True)
    solve_time = time.time() - t_st

    # LIM1TR timing statistics
    if verbose:
        eqn_sys.print_statistics()

    print(f'Total Solve Time (s): {solve_time:0.3f}')

    return eqn_sys.t, q
