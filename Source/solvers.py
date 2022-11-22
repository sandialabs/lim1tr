########################################################################################
#                                                                                      #
#  Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).  #
#  Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains    #
#  certain rights in this software.                                                    #
#                                                                                      #
#  This software is released under the license detailed in the file, LICENSE.          #
#                                                                                      #
########################################################################################

from __future__ import division
import numpy as np
from numba import jit


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


def steady_solve():
    # Provide options for steady spitfire solve
    return 0


def transient_solve(eqn_sys, verbose=True):
    '''Transient solve using Spitfire
    '''
    if eqn_sys.fixed_step:
        step_size = eqn_sys.dt
    else:
        step_size = PIController(target_error=1.e-8)

    t, q = odesolve(self.right_hand_side,
                    t_int.T_star,
                    stop_at_time=t_int.end_time,
                    save_each_step=True,
                    linear_setup=self.setup_superlu,
                    linear_solve=self.solve_superlu,
                    step_size=step_size,
                    linear_setup_rate=20,
                    verbose=verbose,
                    log_rate=100,
                    show_solver_stats_in_situ=True)

    return t, q