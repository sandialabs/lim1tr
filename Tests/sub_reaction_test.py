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
sys.path.append('../Source')
import reaction_models_included
import reaction_model_factory
import reaction_submodels
import unit_mocks


class sub_reaction_tests(unittest.TestCase):
    def test_concentration_product_rule(self):
        print('\nTesting reaction product rule application class on concentration...')
        # Create the base reaction
        rxn_info = unit_mocks.basic_rxn_info_mock
        material_info = unit_mocks.material_info_mock
        main_rxn = reaction_models_included.basic_rxn(rxn_info, material_info)

        # Create the sub reaction
        rxn_info['Electrolyte Limiter'] = {
            'Species': 'R2',
            'Limiting Constant': 2.0}
        sub_rxn = reaction_submodels.electrolyte_limiter(main_rxn)

        # Check concentration outputs
        my_v = np.array([500, 100, 0, 1400, 300])
        f_c_m = my_v[0]
        df_c_m = np.array([1, 0, 0, 0])
        err = np.abs(main_rxn.concentration_function(my_v) - f_c_m)
        err += np.sum(np.abs(main_rxn.concentration_derivative(my_v) - df_c_m))

        f_c_s = my_v[1]/(2.0 + my_v[1])
        df_c_s = np.zeros(4)
        df_c_s[1] = 2.0/(2.0 + my_v[1])**2
        err += np.abs(sub_rxn.concentration_function(my_v) - f_c_s)
        err += np.sum(np.abs(sub_rxn.concentration_derivative(my_v) - df_c_s))

        f_c = f_c_m*f_c_s
        df_c = np.zeros(4)
        df_c[0] = f_c_s*df_c_m[0]
        df_c[1] = f_c_m*df_c_s[1]
        my_rxn = reaction_model_factory.model_chain([main_rxn, sub_rxn])
        err += (my_rxn.concentration_function(my_v) - f_c)
        err += np.sum(np.abs(my_rxn.concentration_derivative(my_v) - df_c))
        self.assertTrue(err < 1e-15, '\tFailed: incorrect concentration computation.\n')


    def test_rate_constant_product_rule_ec(self):
        print('\nTesting reaction product rule application class on a simple rate function...')
        # Create the base reaction
        rxn_info = unit_mocks.basic_rxn_info_mock
        material_info = unit_mocks.material_info_mock
        main_rxn = reaction_models_included.basic_rxn(rxn_info, material_info)

        # Create the sub reaction
        rxn_info['Electrolyte Limiter'] = {
            'Species': 'R2',
            'Limiting Constant': 2.0}
        sub_rxn = reaction_submodels.electrolyte_limiter(main_rxn)

        my_rxn = reaction_model_factory.model_chain([main_rxn, sub_rxn])

        my_v = np.array([500, 100, 0, 1400, 1000])
        my_k_true = 1.0e+9*np.exp(-100000/(8.314*my_v[-1]))
        my_k_dT = my_k_true*(100000/(8.314*my_v[-1]**2))
        err = np.abs(main_rxn.evaluate_rate_constant(my_v) - my_k_true)
        err += np.abs(main_rxn.evaluate_rate_constant_derivative(my_v, my_k_true) - my_k_dT)

        err += np.abs(sub_rxn.evaluate_rate_constant(my_v) - 1)
        err += np.abs(sub_rxn.evaluate_rate_constant_derivative(my_v, 1) - 0)

        err += np.abs(my_rxn.evaluate_rate_constant(my_v) - my_k_true)
        err += np.abs(my_rxn.evaluate_rate_constant_derivative(my_v, my_k_true) - my_k_dT)
        self.assertTrue(err < 1e-15, '\tFailed: incorrect rate constant computation.\n')


    def test_rate_constant_product_rule_exp(self):
        print('\nTesting reaction product rule application class on an exponential rate function...')
        # Create the base reaction
        rxn_info = unit_mocks.basic_rxn_info_mock
        material_info = unit_mocks.material_info_mock
        main_rxn = reaction_models_included.basic_rxn(rxn_info, material_info)

        rxn_info_sub = copy.deepcopy(rxn_info)
        rxn_info_sub['A'] = 50.0
        rxn_info_sub['E'] = 15000.0
        rxn_info_sub['R'] = 1.0
        sub_rxn = reaction_models_included.basic_rxn(rxn_info_sub, material_info)
        my_rxn = reaction_model_factory.model_chain([main_rxn, sub_rxn])

        my_v = np.array([500, 100, 0, 1400, 1000])
        my_k_true = 1.0e+9*np.exp(-100000/(8.314*my_v[-1]))
        my_k_dT = my_k_true*(100000/(8.314*my_v[-1]**2))
        sub_k = 50.0*np.exp(-15000/my_v[-1])
        sub_k_dT = sub_k*15000/my_v[-1]**2
        err = np.abs(sub_rxn.evaluate_rate_constant(my_v) - sub_k)
        err += np.abs(sub_rxn.evaluate_rate_constant_derivative(my_v, sub_k) - sub_k_dT)

        err += np.abs(my_rxn.evaluate_rate_constant(my_v) - my_k_true*sub_k)
        full_der = my_k_true*sub_k_dT + my_k_dT*sub_k
        err += np.abs(my_rxn.evaluate_rate_constant_derivative(my_v, my_k_true*sub_k) - full_der)
        self.assertTrue(err < 1e-15, '\tFailed: incorrect rate constant computation.\n')


if __name__ == '__main__':
    unittest.main()
