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
import time
import sys, os

# Get absolute path of main
main_path = os.path.dirname(os.path.realpath(__file__))

# Import source files
sys.path.append(main_path + '/Source')
import solvers
import input_parser
import build_sys
import equation_sys
import time_integrator


class lim1tr_model:
    def __init__(self, file_name):
        # Print copyright statement
        print('Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).')
        print('Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains')
        print('certain rights in this software.')

        self.parser = input_parser.input_parser(file_name)


    def run_model(self):
        mat_man, grid_man, bc_man, reac_man, data_man, time_opts = self.parser.apply_parse()

        # Evaluate properties
        mat_man.eval_props()

        # Initialize conduction manager
        cond_man = build_sys.conduction_manager(grid_man)

        # Initialize equation system
        eqn_sys = equation_sys.eqn_sys(grid_man, reac_man, time_opts['Solution Mode'], time_opts['Order'])

        # Initialize linear solver (for numba)
        eqn_sys.init_linear_solver()

        # Initialize time integrator
        t_int = time_integrator.time_int(grid_man, time_opts)

        # Solve system
        eqn_sys.solve(mat_man, cond_man, bc_man, reac_man, data_man, t_int)

        # Return managers and options
        return eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, time_opts


if __name__ == '__main__':
    # User supplied file name
    file_name = sys.argv[1]

    # Run
    model = lim1tr_model(file_name)
    model.run_model()
