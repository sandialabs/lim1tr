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
sys.path.append('../')
import main_fv


class mass_loss_tests(unittest.TestCase):
    def test_species_inds(self):
        print('\nTesting solid/gas species indexing...')
        model = main_fv.lim1tr_model('./Inputs/damkohler_anode.yaml')
        model.parser.cap_dict['Species']['Gas Species'] = ['C2H4']
        mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.parser.apply_parse()
        gas_i = reac_man.gas_inds[0]
        gas_name = reac_man.species_name_list[gas_i]
        self.assertTrue(gas_name == 'C2H4', '\tC2H4 not found in species list\n')
        self.assertTrue(len(reac_man.solid_inds) == 5, '\tIncorrect number of solid species\n')


    def test_gas_mass_loss(self):
        print('\nTesting mass loss to a gas species...')
        model = main_fv.lim1tr_model('./Inputs/mass_loss.yaml')
        eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()
        self.assertAlmostEqual(1999.8, data_man.data_dict['A_Solid'][-1,0])
        self.assertAlmostEqual(1999.8, mat_man.rho_arr[-1])


if __name__ == '__main__':
    unittest.main()
