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

from boundary import bc_manager
from boundary_factory import factory


class multiple_bcs_integration_test(unittest.TestCase):
    def setUp(self):
        # Create a mock grid manager
        class MockGridManager:
            def __init__(self):
                self.dx_arr = np.array([0.1, 0.1, 0.1])
                self.n_tot = 3
                self.PA_r = 1.0
                self.mint_list = []

        self.grid_man = MockGridManager()
        self.bc_man = bc_manager(self.grid_man)


    def test_multiple_bcs_setup(self):
        '''Test that the boundary manager can handle multiple BCs per face.'''
        # Create boundary dictionary with multiple BCs on External face
        bnd_dict = {
            'External': [
                {
                    'Type': 'convection',
                    'h': 10.0,
                    'T': 298.15
                },
                {
                    'Type': 'radiation',
                    'eps': 0.8,
                    'T': 298.15
                }
            ],
            'Left': {
                'Type': 'adiabatic'
            },
            'Right': {
                'Type': 'adiabatic'
            }
        }

        # This should not raise an error
        self.bc_man.setup(bnd_dict)

        # Should have registered 4 BCs total (2 external + 1 left adiabatic + 1 right adiabatic)
        total_bcs = len(self.bc_man.boundaries) + len(self.bc_man.nonlinear_boundaries)
        self.assertEqual(total_bcs, 4)  # convection, radiation, left adiabatic, right adiabatic

        # Convection should be linear, radiation should be nonlinear, adiabatic are linear
        self.assertEqual(len(self.bc_man.boundaries), 3)  # convection + 2 adiabatic
        self.assertEqual(len(self.bc_man.nonlinear_boundaries), 1)  # radiation
        self.assertTrue(self.bc_man.nonlinear_flag)


    def test_backward_compatibility(self):
        '''Test that single BC specifications still work.'''
        # Create boundary dictionary with single BCs (old format)
        bnd_dict = {
            'External': {
                'Type': 'convection',
                'h': 15.0,
                'T': 350.0
            },
            'Left': {
                'Type': 'adiabatic'
            },
            'Right': {
                'Type': 'adiabatic'
            }
        }

        # This should not raise an error
        self.bc_man.setup(bnd_dict)

        # Should have registered 3 BCs (convection + 2 adiabatic)
        total_bcs = len(self.bc_man.boundaries) + len(self.bc_man.nonlinear_boundaries)
        self.assertEqual(total_bcs, 3)
        self.assertEqual(len(self.bc_man.boundaries), 3)  # convection + 2 adiabatic
        self.assertEqual(len(self.bc_man.nonlinear_boundaries), 0)
        self.assertFalse(self.bc_man.nonlinear_flag)


    def test_empty_bc_list_defaults_to_adiabatic(self):
        '''Test that empty BC list defaults to adiabatic.'''
        # Create boundary dictionary with empty BC list
        bnd_dict = {
            'External': [],
            'Left': {
                'Type': 'adiabatic'
            },
            'Right': {
                'Type': 'adiabatic'
            }
        }

        # This should not raise an error and should default to adiabatic
        self.bc_man.setup(bnd_dict)

        # Adiabatic BCs do get registered but don't modify the system
        total_bcs = len(self.bc_man.boundaries) + len(self.bc_man.nonlinear_boundaries)
        self.assertEqual(total_bcs, 3)  # 3 adiabatic BCs (external, left, and right)


if __name__ == '__main__':
    unittest.main()
