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

from boundary_factory import factory


class external_spatial_bc_test(unittest.TestCase):
    def setUp(self):
        self.dx_arr = np.array([0.1, 0.1, 0.1])
        self.x_node = np.array([0.05, 0.15, 0.25])
        self.PA_r = 10.0
        self.mint_list = []
        self.x_table_path = os.path.join(os.path.dirname(__file__), 'Inputs', 'x_profile.csv')
        self.xt_table_path = os.path.join(os.path.dirname(__file__), 'Inputs', 'xt_profile.csv')


    def test_x_table_convection_h(self):
        '''X Table should linearly interpolate h onto x_node.'''
        params = {
            'Type': 'Convection',
            'h': {'X Table': self.x_table_path},
            'T': 300.0
        }
        bcs = factory('External', params, self.dx_arr, self.PA_r, self.mint_list, x_node=self.x_node)
        expected_h = np.interp(self.x_node, [0, 0.5, 1], [10, 20, 30])
        np.testing.assert_allclose(bcs[0].h, expected_h)


    def test_x_table_radiation_eps(self):
        '''X Table (static) is allowed on radiation since it bypasses the temporal wrapper.'''
        params = {
            'Type': 'Radiation',
            'eps': {'X Table': self.x_table_path},
            'T': 300.0
        }
        bcs = factory('External', params, self.dx_arr, self.PA_r, self.mint_list, x_node=self.x_node)
        expected_eps = np.interp(self.x_node, [0, 0.5, 1], [10, 20, 30])
        np.testing.assert_allclose(bcs[0].eps, expected_eps)


    def test_xt_table_convection_h_at_multiple_times(self):
        '''XT Table should bilinearly interpolate h over x_node as t advances.'''
        params = {
            'Type': 'Convection',
            'h': {'XT Table': self.xt_table_path},
            'T': 300.0
        }
        bcs = factory('External', params, self.dx_arr, self.PA_r, self.mint_list, x_node=self.x_node)
        bc = bcs[0]

        # initial value (t=0) set directly on the wrapped bc by the factory
        np.testing.assert_allclose(bc.bc.h, [11, 13, 15], atol=1e-8)

        # advancing time should update h via the temporal wrapper
        bc.update_params(5.0)
        np.testing.assert_allclose(bc.bc.h, [16, 18, 20], atol=1e-8)

        bc.update_params(10.0)
        np.testing.assert_allclose(bc.bc.h, [21, 23, 25], atol=1e-8)


    def test_x_table_rejected_on_end_bc(self):
        params = {
            'Type': 'Convection',
            'h': {'X Table': self.x_table_path},
            'T': 300.0
        }
        with self.assertRaises(ValueError):
            factory('Left', params, self.dx_arr, self.PA_r, self.mint_list, x_node=self.x_node)


    def test_xt_table_rejected_on_end_bc(self):
        params = {
            'Type': 'Convection',
            'h': {'XT Table': self.xt_table_path},
            'T': 300.0
        }
        with self.assertRaises(ValueError):
            factory('Right', params, self.dx_arr, self.PA_r, self.mint_list, x_node=self.x_node)


    def test_xt_table_rejected_on_radiation(self):
        '''XT Table requires the temporal wrapper, which radiation does not support.'''
        params = {
            'Type': 'Radiation',
            'eps': {'XT Table': self.xt_table_path},
            'T': 300.0
        }
        with self.assertRaises(ValueError):
            factory('External', params, self.dx_arr, self.PA_r, self.mint_list, x_node=self.x_node)


if __name__ == '__main__':
    unittest.main()
