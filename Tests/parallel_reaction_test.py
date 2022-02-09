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
import unittest
import numpy as np
import time, sys, os, copy
import multiprocessing as mp
sys.path.append('../Source')
import reaction


class parallel_reaction_mock(reaction.reaction_manager):
    def __init__(self):
        self.inactive_node_list = []
        self.active_nodes = None
        self.solve_ode_time = 0.0
        self.update_dofs_time = 0.0


    def solve_nodes_wrapper(self, my_active_node_inds):
        sol_err_list = []
        for act_ind in my_active_node_inds:
            i = self.active_nodes[act_ind]
            my_sol = np.zeros([1,1]) + i
            sol_err_list.append((my_sol, 0))
        return sol_err_list


    def update_node(self, act_ind, my_ind):
        return 0.0


class parallel_reaction_tests(unittest.TestCase):
    def test_parallel_node_map(self):
        print('\nTesting parallel ODE solves...')
        if sys.version_info[0] < 3:
            print('Python 2 detected, skipping parallel tests...')
            return 0

        reac_man = parallel_reaction_mock()
        n_nodes = 10

        # Solution should come back in order
        T_true = np.arange(n_nodes)

        # Serial
        reac_man.active_nodes = np.arange(n_nodes)
        T_in = np.zeros(n_nodes)
        T_out, err_list = reac_man.solve_ode_all_nodes(0.0, T_in, pool=None, n_cores=1)
        err = np.mean((T_out - T_true)**2)**(0.5)

        # 2 Core
        pool = mp.Pool(2)
        T_in = np.zeros(n_nodes)
        T_out, err_list = reac_man.solve_ode_all_nodes(0.0, T_in, pool=pool, n_cores=2)
        err += np.mean((T_out - T_true)**2)**(0.5)

        # 2 Core node index scramble
        reac_man.active_nodes = np.array([1, 3, 4, 7, 8, 0, 9, 6, 5, 2])
        T_in = np.zeros(n_nodes)
        T_out, err_list = reac_man.solve_ode_all_nodes(0.0, T_in, pool=pool, n_cores=2)
        err += np.mean((T_out - T_true)**2)**(0.5)
        pool.close()

        # 4 Core node index scramble
        pool = mp.Pool(4)
        reac_man.active_nodes = np.array([1, 3, 4, 7, 8, 0, 9, 6, 5, 2])
        T_in = np.zeros(n_nodes)
        T_out, err_list = reac_man.solve_ode_all_nodes(0.0, T_in, pool=pool, n_cores=4)
        err += np.mean((T_out - T_true)**2)**(0.5)
        pool.close()

        self.assertTrue(err < 1e-15, '\tFailed with RMSE {:0.2e}\n'.format(err))


if __name__ == '__main__':
    unittest.main()
