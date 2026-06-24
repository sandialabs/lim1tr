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
import os

# Add the Source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../Source'))

from boundary_types import end_convection, end_radiation
from boundary_factory import factory


class multiple_bcs_unit_test(unittest.TestCase):
    def setUp(self):
        self.dx_arr = np.array([0.1, 0.1, 0.1])
        self.PA_r = 1.0
        self.mint_list = []


    def test_single_bc_factory(self):
        '''Test that single BC creation still works (backward compatibility).'''
        params = {
            'Type': 'Convection',
            'h': 10.0,
            'T': 300.0
        }

        bcs = factory('Left', params, self.dx_arr, self.PA_r, self.mint_list)

        # Should return a list with one BC
        self.assertIsInstance(bcs, list)
        self.assertEqual(len(bcs), 1)
        self.assertIsInstance(bcs[0], end_convection)
        self.assertEqual(bcs[0].h, 10.0)
        self.assertEqual(bcs[0].T, 300.0)


    def test_multiple_bcs_factory(self):
        '''Test that multiple BCs create a list of BC objects.'''
        params = [
            {
                'Type': 'Convection',
                'h': 10.0,
                'T': 300.0
            },
            {
                'Type': 'Radiation',
                'eps': 0.8,
                'T': 298.0
            }
        ]

        bcs = factory('Left', params, self.dx_arr, self.PA_r, self.mint_list)

        # Should return a list with two BCs
        self.assertIsInstance(bcs, list)
        self.assertEqual(len(bcs), 2)
        self.assertIsInstance(bcs[0], end_convection)
        self.assertIsInstance(bcs[1], end_radiation)


    def test_empty_params_defaults_to_adiabatic(self):
        '''Test that empty parameters default to adiabatic BC.'''
        bcs = factory('Left', [], self.dx_arr, self.PA_r, self.mint_list)

        # Should return a list with one adiabatic BC
        self.assertIsInstance(bcs, list)
        self.assertEqual(len(bcs), 1)


    def test_is_linear_property(self):
        '''Test that is_linear property works correctly.'''
        conv_bc = end_convection(self.dx_arr, 'Left')
        rad_bc = end_radiation(self.dx_arr, 'Left')

        # Convection should be linear
        self.assertTrue(conv_bc.is_linear)

        # Radiation should be nonlinear
        self.assertFalse(rad_bc.is_linear)


if __name__ == '__main__':
    unittest.main()
