########################################################################################
#                                                                                      #
#  Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).  #
#  Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains    #
#  certain rights in this software.                                                    #
#                                                                                      #
#  This software is released under the license detailed in the file, LICENSE.          #
#                                                                                      #
########################################################################################

import unittest
import numpy as np
import time, sys, os, copy
sys.path.append('../Source')
import reaction_system_helper


class reaction_manager_tests(unittest.TestCase):
    def verify_unique_system_test(self, true_index, true_list, reac_index, reac_list):
        test_passed = True
        err_message = ''
        for i in range(true_index.shape[0]):
            if true_index[i] != reac_index[i]:
                test_passed = False
                err_message = '\tFailed: reaction system index incorrect.\n'
                return test_passed, err_message
        if len(true_list) != len(reac_list):
            test_passed = False
            err_message = '\tFailed: incorrect number of unique reactions found.\n'
            return test_passed, err_message
        for i in range(len(true_list)):
            for j in range(true_list[i].shape[0]):
                if true_list[i][j] != reac_list[i][j]:
                    test_passed = False
                    err_message = '\tFailed: incorrect unique reaction list.\n'
                    return test_passed, err_message

        return test_passed, err_message


    def test_one_unique_system(self):
        print('\nTesting unique reaction system identification for 1 reaction system...')
        n_rxn = 3
        n_cells = 5
        active_cells = np.ones([n_rxn, n_cells], dtype=int)
        system_index, unique_system_list = reaction_system_helper.find_unique_systems(active_cells)
        true_system_index = np.zeros(n_cells, dtype=int)
        true_unique_system_list = [np.ones(n_rxn, dtype=int)]

        test_passed, err_message = self.verify_unique_system_test(true_system_index,
            true_unique_system_list, system_index, unique_system_list)
        self.assertTrue(test_passed, err_message)


    def test_two_unique_system(self):
        print('\nTesting unique reaction system identification for 2 reaction systems...')
        n_rxn = 3
        n_cells = 5
        active_cells = np.ones([n_rxn, n_cells], dtype=int)
        active_cells[:,0] = [1,1,0]
        active_cells[:,3] = [1,1,0]
        system_index, unique_system_list = reaction_system_helper.find_unique_systems(active_cells)
        true_system_index = np.array([0,1,1,0,1], dtype=int)
        true_unique_system_list = [np.array([1,1,0], dtype=int),
            np.ones(n_rxn, dtype=int)]

        test_passed, err_message = self.verify_unique_system_test(true_system_index,
            true_unique_system_list, system_index, unique_system_list)
        self.assertTrue(test_passed, err_message)


    def test_three_unique_system(self):
        print('\nTesting unique reaction system identification for 3 reaction systems...')
        n_rxn = 3
        n_cells = 5
        active_cells = np.ones([n_rxn, n_cells], dtype=int)
        active_cells[:,0] = [1,1,0]
        active_cells[:,3] = [1,1,0]
        active_cells[:,4] = [0,0,1]
        system_index, unique_system_list = reaction_system_helper.find_unique_systems(active_cells)
        true_system_index = np.array([0,1,1,0,2], dtype=int)
        true_unique_system_list = [np.array([1,1,0], dtype=int),
            np.ones(n_rxn, dtype=int),
            np.array([0,0,1], dtype=int)]

        test_passed, err_message = self.verify_unique_system_test(true_system_index,
            true_unique_system_list, system_index, unique_system_list)
        self.assertTrue(test_passed, err_message)


if __name__ == '__main__':
    unittest.main()
