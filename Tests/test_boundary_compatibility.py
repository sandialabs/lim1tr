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
import sys
import os
import numpy as np

# Add the lim1tr Source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../Source'))

import boundary_factory


class TestBoundaryCompatibility(unittest.TestCase):
    def setUp(self):
        '''Set up test fixtures.'''
        self.dx_arr = np.array([0.1, 0.1, 0.1])
        self.PA_r = 1.0
        self.mint_list = [0, 1, 2]


    def test_valid_convection_radiation_combination(self):
        '''Test that convection + radiation combination is valid.'''
        params = [
            {'Type': 'convection', 'h': 10.0, 'T': 298.15},
            {'Type': 'radiation', 'eps': 0.8, 'T': 298.15}
        ]

        # This should not raise an exception
        bc_objects = boundary_factory.factory('External', params, self.dx_arr, self.PA_r, self.mint_list)
        self.assertEqual(len(bc_objects), 2)


    def test_valid_single_adiabatic(self):
        '''Test that single adiabatic BC is valid.'''
        params = [{'Type': 'adiabatic'}]

        # This should not raise an exception
        bc_objects = boundary_factory.factory('Left', params, self.dx_arr, self.PA_r, self.mint_list)
        self.assertEqual(len(bc_objects), 1)


    def test_valid_single_dirichlet(self):
        '''Test that single dirichlet BC is valid.'''
        params = [{'Type': 'dirichlet', 'T': 300.0}]

        # This should not raise an exception
        bc_objects = boundary_factory.factory('Right', params, self.dx_arr, self.PA_r, self.mint_list)
        self.assertEqual(len(bc_objects), 1)


    def test_invalid_adiabatic_convection_combination(self):
        '''Test that adiabatic + convection combination is invalid.'''
        params = [
            {'Type': 'adiabatic'},
            {'Type': 'convection', 'h': 10.0, 'T': 298.15}
        ]

        with self.assertRaises(ValueError) as context:
            boundary_factory.factory('External', params, self.dx_arr, self.PA_r, self.mint_list)

        self.assertIn('Cannot combine', str(context.exception))
        self.assertIn('adiabatic', str(context.exception))


    def test_invalid_dirichlet_flux_combination(self):
        '''Test that dirichlet + flux combination is invalid.'''
        params = [
            {'Type': 'dirichlet', 'T': 300.0},
            {'Type': 'heat flux', 'Flux': 100.0}
        ]

        with self.assertRaises(ValueError) as context:
            boundary_factory.factory('Left', params, self.dx_arr, self.PA_r, self.mint_list)

        self.assertIn('Cannot combine', str(context.exception))
        self.assertIn('dirichlet', str(context.exception))


    def test_invalid_adiabatic_radiation_combination(self):
        '''Test that adiabatic + radiation combination is invalid.'''
        params = [
            {'Type': 'adiabatic'},
            {'Type': 'radiation', 'eps': 0.8, 'T': 298.15}
        ]

        with self.assertRaises(ValueError) as context:
            boundary_factory.factory('External', params, self.dx_arr, self.PA_r, self.mint_list)

        self.assertIn('Cannot combine', str(context.exception))
        self.assertIn('adiabatic', str(context.exception))


    def test_valid_multiple_convection(self):
        '''Test that multiple convection BCs are valid (same type).'''
        params = [
            {'Type': 'convection', 'h': 10.0, 'T': 298.15},
            {'Type': 'convection', 'h': 15.0, 'T': 300.0}
        ]

        # This should not raise an exception - multiple same-type BCs should be allowed
        bc_objects = boundary_factory.factory('External', params, self.dx_arr, self.PA_r, self.mint_list)
        self.assertEqual(len(bc_objects), 2)


    def test_valid_convection_flux_combination(self):
        '''Test that convection + flux combination is valid.'''
        params = [
            {'Type': 'convection', 'h': 10.0, 'T': 298.15},
            {'Type': 'heat flux', 'Flux': 100.0}
        ]

        # This should not raise an exception (using Left boundary since heat flux is only for end boundaries)
        bc_objects = boundary_factory.factory('Left', params, self.dx_arr, self.PA_r, self.mint_list)
        self.assertEqual(len(bc_objects), 2)


    def test_invalid_multiple_adiabatic_combination(self):
        '''Test that multiple adiabatic BCs are invalid.'''
        params = [
            {'Type': 'adiabatic'},
            {'Type': 'adiabatic'}
        ]

        with self.assertRaises(ValueError) as context:
            boundary_factory.factory('Left', params, self.dx_arr, self.PA_r, self.mint_list)

        self.assertIn('Cannot combine', str(context.exception))
        self.assertIn('adiabatic', str(context.exception))


if __name__ == '__main__':
    unittest.main()