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

from boundary_factory import compute_face_PA_r, factory


class external_faces_test(unittest.TestCase):
    def setUp(self):
        self.L_y = 2.0
        self.L_z = 1.0
        self.dx_arr = np.array([0.1, 0.1, 0.1])
        self.mint_list = []


    def test_single_face(self):
        '''Top/Bottom span Y Dimension, so each contributes 1/L_z.'''
        self.assertAlmostEqual(compute_face_PA_r(['Top'], self.L_y, self.L_z), 1./self.L_z)
        self.assertAlmostEqual(compute_face_PA_r(['Bottom'], self.L_y, self.L_z), 1./self.L_z)


    def test_opposite_pair(self):
        '''Front/Back span Z Dimension, so each contributes 1/L_y.'''
        pa_r = compute_face_PA_r(['Front', 'Back'], self.L_y, self.L_z)
        self.assertAlmostEqual(pa_r, 2./self.L_y)


    def test_all_four_faces_match_legacy_formula(self):
        pa_r = compute_face_PA_r(['Top', 'Bottom', 'Front', 'Back'], self.L_y, self.L_z)
        legacy_pa_r = 2.*(self.L_y + self.L_z)/(self.L_y*self.L_z)
        self.assertAlmostEqual(pa_r, legacy_pa_r)


    def test_case_insensitive(self):
        pa_r = compute_face_PA_r(['top', 'BOTTOM'], self.L_y, self.L_z)
        self.assertAlmostEqual(pa_r, 2./self.L_z)


    def test_unrecognized_face_raises(self):
        with self.assertRaises(ValueError):
            compute_face_PA_r(['Side'], self.L_y, self.L_z)


    def test_duplicate_face_raises(self):
        with self.assertRaises(ValueError):
            compute_face_PA_r(['Top', 'Top'], self.L_y, self.L_z)


    def test_factory_uses_faces_subset(self):
        '''An External BC with Faces specified should get a partial PA_r.'''
        params = {
            'Type': 'Convection',
            'h': 10.0,
            'T': 300.0,
            'Faces': ['Top', 'Bottom']
        }

        full_pa_r = 2.*(self.L_y + self.L_z)/(self.L_y*self.L_z)
        bcs = factory('External', params, self.dx_arr, full_pa_r, self.mint_list, self.L_y, self.L_z)

        expected_pa_r = compute_face_PA_r(['Top', 'Bottom'], self.L_y, self.L_z)
        np.testing.assert_allclose(bcs[0].dx_PA_r, self.dx_arr*expected_pa_r)


    def test_factory_without_faces_uses_full_PA_r(self):
        '''Omitting Faces should fall back to the full PA_r passed in (legacy behavior).'''
        params = {
            'Type': 'Convection',
            'h': 10.0,
            'T': 300.0
        }

        full_pa_r = 2.*(self.L_y + self.L_z)/(self.L_y*self.L_z)
        bcs = factory('External', params, self.dx_arr, full_pa_r, self.mint_list, self.L_y, self.L_z)

        np.testing.assert_allclose(bcs[0].dx_PA_r, self.dx_arr*full_pa_r)


if __name__ == '__main__':
    unittest.main()
