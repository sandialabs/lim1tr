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
import sys
import unit_meshes


class material_tests(unittest.TestCase):
    def test_k_interface(self):
        print('\nTesting k at interface...')
        k_a = 5.
        k_b = 15.
        dx_a = 0.5
        dx_b = 0.25

        grid_man, mat_man = unit_meshes.mesh_two(dx_a=dx_a,k_a=k_a,dx_b=dx_b,k_b=k_b)

        # True k array
        k_true = np.zeros(9)
        for i in range(9):
            if i < grid_man.mint_list[0]:
                k_true[i] = k_a
            elif i == grid_man.mint_list[0]:
                k_true[i] = (dx_a + dx_b)*k_a*k_b/(k_b*dx_a + k_a*dx_b)
            elif i > grid_man.mint_list[0]:
                k_true[i] = k_b

        # Compare
        err = np.sum((mat_man.k_arr-k_true)**2)
        self.assertTrue(err < 1e-12, '\tFailed with error {:0.1e}\n'.format(err))


if __name__ == '__main__':
    # Run tests
    unittest.main()
