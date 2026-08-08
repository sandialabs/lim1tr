########################################################################################
#                                                                                      #
#  Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).  #
#  Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains    #
#  certain rights in this software.                                                    #
#                                                                                      #
#  This software is released under the license detailed in the file, LICENSE.          #
#                                                                                      #
########################################################################################

import sys
import time
import logging
from contextlib import redirect_stdout, nullcontext
from spitfire import PIController, odesolve
from spitfire import SimpleNewtonSolver, KennedyCarpenterS6P4Q3


class _Tee:
    '''Duplicates writes to an underlying stream and a log file.

    Spitfire's odesolve writes its verbose solver progress with raw
    print() calls rather than the logging module, so capturing it in
    the log file requires redirecting stdout during the solve.
    '''
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


def _tee_to_log_file():
    '''Redirect stdout to the active log FileHandler (if any) as well as the console.'''
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            return redirect_stdout(_Tee(sys.stdout, handler.stream))
    return nullcontext()


def steady_solve(eqn_sys):
    eqn_sys.steady_solve()
    logging.info('Conduction Solve Time: {:0.2f} s'.format(eqn_sys.time_conduction))


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
    method=KennedyCarpenterS6P4Q3(SimpleNewtonSolver(norm_weighting=eqn_sys.norm_weighting))

    t_st = time.time()
    with _tee_to_log_file():
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

    logging.info(f'Total Solve Time (s): {solve_time:0.3f}')

    return eqn_sys.t, q
