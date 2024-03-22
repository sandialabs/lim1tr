########################################################################################
#                                                                                      #
#  Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).  #
#  Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains    #
#  certain rights in this software.                                                    #
#                                                                                      #
#  This software is released under the license detailed in the file, LICENSE.          #
#                                                                                      #
########################################################################################

import time
from spitfire import PIController, odesolve
from spitfire import SimpleNewtonSolver, KennedyCarpenterS6P4Q3


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

    if eqn_sys.reac_man:
        linear_setup=eqn_sys.setup_superlu
        linear_solve=eqn_sys.solve_superlu
    else:
        linear_setup=eqn_sys.setup_conduction
        linear_solve=eqn_sys.solve_conduction
    method=KennedyCarpenterS6P4Q3(SimpleNewtonSolver())

    t_st = time.time()
    q = odesolve(eqn_sys.right_hand_side,
                 eqn_sys.initial_state,
                 output_times=eqn_sys.t,
                 linear_setup=linear_setup,
                 linear_solve=linear_solve,
                 norm_weighting=eqn_sys.norm_weighting,
                 post_step_callback=eqn_sys.post_step,
                 method=method,
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
