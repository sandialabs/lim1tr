########################################################################################
#                                                                                      #
#  Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).  #
#  Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains    #
#  certain rights in this software.                                                    #
#                                                                                      #
#  This software is released under the license detailed in the file, LICENSE.          #
#                                                                                      #
########################################################################################

import sys, os

# Get absolute path of main
main_path = os.path.dirname(os.path.realpath(__file__))

# Import source files
sys.path.append(main_path + '/Source')
import solvers
import input_parser
import conduction
import equation_sys


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
        cond_man = conduction.conduction_manager(grid_man)

        # Initialize equation system
        eqn_sys = equation_sys.eqn_sys(mat_man, cond_man, bc_man, grid_man, reac_man, time_opts)

        # Solve system
        if 'Steady' in time_opts['Solution Mode']:
            solvers.steady_solve(eqn_sys)
        else:
            t, q = solvers.transient_solve(eqn_sys, verbose=time_opts['Print Progress'])
            data_man.format_data(t, q)
            data_man.write_data()

        # Return managers and options
        return eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts


if __name__ == '__main__':
    # User supplied file name
    file_name = sys.argv[1]

    # Run
    model = lim1tr_model(file_name)
    model.run_model()
