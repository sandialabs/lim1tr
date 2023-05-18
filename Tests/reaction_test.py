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
from scipy.special import expi
sys.path.append('../')
import main_fv
sys.path.append('../Source')
import input_parser
import reaction

import matplotlib as mpl
mpl.use( 'Agg' )
import matplotlib.pyplot as plt


class reaction_tests(unittest.TestCase):
    def setUp(self):
        self.plotting = False
        self.grad_check = False


    def test_single_rxn_temperature_ramp(self):
        print('\nTesting single reaction with a constant temperature ramp...')

        # Solve reaction system
        model = main_fv.lim1tr_model('./Inputs/single_rxn_temperature_ramp.yaml')
        eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()
        sol = data_man.data_dict['AA']
        t_arr = eqn_sys.t

        # Analytical solution
        T_rate = 5. # C/s
        T_i = 298.15
        A = 10.
        EoR = 1000.
        rho_i = 10.
        T_arr = T_rate*t_arr + T_i
        expr_A = T_arr*np.exp(-EoR/T_arr) - T_i*np.exp(-EoR/T_i)
        expr_B = EoR*(expi(-EoR/T_arr) - expi(-EoR/T_i))
        rho_arr = rho_i*np.exp(-A*(expr_A + expr_B)/T_rate)

        # Calculate error
        err = np.sum((rho_arr - sol[:,0])**2/t_arr.shape[0])**0.5

        if self.plotting:
            plt.figure()
            plt.plot(t_arr, rho_arr, 'o', label='Analytical AA')
            for i in range(reac_man.n_species):
                plt.plot(t_arr, sol[:,i], '-', label='Numerical ' + reac_man.species_name_list[i])
            plt.xlabel(r'Time ($s$)')
            plt.ylabel(r'Density ($kg/m^3$)')
            plt.legend()
            plt.title('RMSE = {:.3E}'.format(err))
            plt.savefig('./Figures/single_rxn_temperature_ramp.png', bbox_inches='tight')
            plt.close()

        self.assertTrue(err < 2e-7, '\tFailed with RMSE {:0.2e}\n'.format(err))


    def test_short_rxn_C6Li(self):
        '''Check the short reaction evaluation at a given state for C6Li limiting
        '''
        print('\nTesting short circuit reaction with C6Li limiting...')
        # Run Model
        model = main_fv.lim1tr_model('./Inputs/short_only.yaml')
        eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()

        # Compute solution
        voltage = 4.2
        short_resistance = 0.001
        volume = 4.8e-5
        charge_kmol = 1000*6.022e23*1.6023e-19
        kg_reactants = 79.007 + 90.931
        Y_o = model.parser.cap_dict['Species']['Initial Mass Fraction']
        rho_C6Li_o = 2000*Y_o[0]
        rho_CoO2_o = 2000*Y_o[1]
        t_arr = np.linspace(0, 4, 41)
        dT_dt = voltage**2/(short_resistance*volume*2000*800)
        rate = voltage*kg_reactants/(short_resistance*charge_kmol*volume)
        rho_C6Li = rho_C6Li_o - rate*t_arr*(79.007/(79.007 + 90.931))
        T_ans = 298.15 + dT_dt*t_arr
        t_complete = rho_C6Li_o/(rate*79.007/(79.007 + 90.931))
        T_f = 298.15 + dT_dt*t_complete
        for i in range(t_arr.shape[0]):
            if rho_C6Li[i] < 0:
                rho_C6Li[i] = 0
                T_ans[i] = T_f

        err = np.sqrt(np.mean((rho_C6Li - data_man.data_dict['C6Li'][:,0])**2))
        err += np.sqrt(np.mean((T_ans - data_man.data_dict['Temperature'][:,0])**2))
        self.assertTrue(err < 1e-5, '\tFailed with RMSE {:0.2e}\n'.format(err))


    def test_short_rxn_CoO2(self):
        '''Check the short reaction evaluation at a given state for CoO2 limiting
        '''
        print('\nTesting short circuit reaction with CoO2 limiting...')
        # Run Model
        model = main_fv.lim1tr_model('./Inputs/short_only.yaml')
        model.parser.cap_dict['Species']['Initial Mass Fraction'][0] = 0.16
        model.parser.cap_dict['Species']['Initial Mass Fraction'][1] = 0.13
        eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()

        # Compute solution
        voltage = 4.2
        short_resistance = 0.001
        volume = 4.8e-5
        charge_kmol = 1000*6.022e23*1.6023e-19
        kg_reactants = 79.007 + 90.931
        Y_o = model.parser.cap_dict['Species']['Initial Mass Fraction']
        rho_CoO2_o = 2000*Y_o[1]
        t_arr = np.linspace(0, 4, 41)
        dT_dt = voltage**2/(short_resistance*volume*2000*800)
        rate = voltage*kg_reactants/(short_resistance*charge_kmol*volume)
        rho_CoO2 = rho_CoO2_o - rate*t_arr*(90.931/(79.007 + 90.931))
        T_ans = 298.15 + dT_dt*t_arr
        t_complete = rho_CoO2_o/(rate*90.931/(79.007 + 90.931))
        T_f = 298.15 + dT_dt*t_complete
        for i in range(t_arr.shape[0]):
            if rho_CoO2[i] < 0:
                rho_CoO2[i] = 0
                T_ans[i] = T_f

        err = np.sqrt(np.mean((rho_CoO2 - data_man.data_dict['CoO2'][:,0])**2))
        err += np.sqrt(np.mean((T_ans - data_man.data_dict['Temperature'][:,0])**2))
        self.assertTrue(err < 1e-5, '\tFailed with RMSE {:0.2e}\n'.format(err))


    def test_zcrit_rxn(self):
        '''Check the zcrit reaction evaluation and Jacobian at a given state
        '''
        print('\nTesting critical thickness anode reaction...')
        # Parse file
        model = main_fv.lim1tr_model('./Inputs/anode_only.yaml')
        mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.parser.apply_parse()

        # Build variable vector
        my_s = np.zeros(reac_man.n_species)
        for i in range(len(reac_man.species_name_list)):
            my_s[i] = reac_man.species_density[reac_man.species_name_list[i]][0]
        my_v = np.zeros(reac_man.n_species + 1)
        my_v[0] = time_opts['T Initial']
        my_v[1:] = my_s

        # Compute function at base inputs
        RHS_T, RHS_species = reac_man.right_hand_side(0, my_v)

        # Compute solution
        reac_sys = reac_man.reaction_systems[0]
        my_rxn = reac_sys.model_list[0].my_funcs[0]
        rho_50 = my_rxn.rxn_info['BET_C6']*my_rxn.material_info['rho']*my_rxn.rxn_info['Y_Graphite']/200.0
        rho_fun = my_s[my_rxn.name_map['C6Li']]*my_s[my_rxn.name_map['EC']]/(rho_50 + my_s[my_rxn.name_map['EC']])
        crit_fun = np.exp(-my_rxn.C_t*my_rxn.z_c*my_s[my_rxn.name_map['Li2CO3']])
        conc_fun_1 = my_rxn.a_e_crit*rho_fun*crit_fun
        r_1 = -1.*my_rxn.A*conc_fun_1*np.exp(-my_rxn.EoR/time_opts['T Initial'])

        err = np.abs(r_1[0]*2*79.007/(2*79.007 + 88.062) - RHS_species[0])
        self.assertTrue(err < 1e-12, '\tFailed with RMSE {:0.2e}\n'.format(err))


    def fd_check(self, file_name, du=1e-5):
        '''Check the Jacobian computation with finite differences
        '''
        print('\nChecking Jacobian computation with finite differences for "{}"...'.format(file_name))
        np.set_printoptions(linewidth = 200)

        # Parse file
        model = main_fv.lim1tr_model('./Inputs/' + file_name + '.yaml')
        mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.parser.apply_parse()

        # Build variable vector
        my_s = np.zeros(reac_man.n_species)
        for i in range(len(reac_man.species_name_list)):
            my_s[i] = reac_man.species_density[reac_man.species_name_list[i]][0]
        my_v = np.zeros(reac_man.n_species + 1)
        my_v[0] = time_opts['T Initial']
        my_v[1:] = my_s
        n_v = reac_man.n_species + 1

        # Compute function and Jacobian at base inputs
        RHS_T, RHS_species = reac_man.right_hand_side(0, my_v)
        my_f = np.hstack([RHS_T, RHS_species])
        R_jac = reac_man.jacobian(0, my_v).ravel().reshape([n_v, n_v])

        # Compute Jacobain from finite differences
        if self.grad_check:
            print('Performing FD gradient check...')
            for up in range(1,11):
                du = 10**(-up)
                jac_fin = np.zeros([n_v, n_v])
                for i in range(my_v.shape[0]):
                    tmp_v = np.copy(my_v)
                    tmp_v[i] += du
                    RHS_T, RHS_species = reac_man.right_hand_side(0, tmp_v)
                    tmp_f = np.hstack([RHS_T, RHS_species])
                    jac_fin[:,i] = (tmp_f - my_f)/du

                # Compute relative error
                rel_err = np.zeros(R_jac.shape)
                N_v = 0
                err = 0.
                for i in range(n_v):
                    for j in range(n_v):
                        if np.abs(R_jac[i,j]) > 1e-15:
                            err_temp = np.abs((R_jac[i,j] - jac_fin[i,j])/R_jac[i,j])
                            rel_err[i,j] = err_temp
                            err += err_temp
                            N_v += 1
                print('Error: {:0.2e}, du: {:0.1e}'.format(err/N_v, du))
            return 0

        else:
            jac_fin = np.zeros([n_v, n_v])
            for i in range(my_v.shape[0]):
                tmp_v = np.copy(my_v)
                tmp_v[i] += du
                RHS_T, RHS_species = reac_man.right_hand_side(0, tmp_v)
                tmp_f = np.hstack([RHS_T, RHS_species])
                jac_fin[:,i] = (tmp_f - my_f)/du

            # Compute relative error
            N_v = 0
            err = 0
            for i in range(n_v):
                for j in range(n_v):
                    if np.abs(R_jac[i,j]) > 1e-15:
                        err += np.abs((R_jac[i,j] - jac_fin[i,j])/R_jac[i,j])
                        N_v += 1
            return err/N_v


    def test_jac(self):
        err = self.fd_check('jac_test', du=1e-2)
        self.assertTrue(err < 1e-4, '\tFailed with Error {:0.2e}\n'.format(err))


    def test_jac_single(self):
        err = self.fd_check('jac_test_single')
        self.assertTrue(err < 1e-7, '\tFailed with Error {:0.2e}\n'.format(err))


    def test_jac_anode(self):
        err = self.fd_check('anode_only', du=1e-6)
        self.assertTrue(err < 3e-7, '\tFailed with Error {:0.2e}\n'.format(err))


    def test_jac_damkohler(self):
        err = self.fd_check('damkohler_anode', du=1e-6)
        self.assertTrue(err < 4e-6, '\tFailed with Error {:0.2e}\n'.format(err))


    def test_damkohler_anode(self):
        print('\nTesting Damkohler limiter on the critical thickness anode...')
        # Parse file
        model = main_fv.lim1tr_model('./Inputs/damkohler_anode.yaml')
        mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.parser.apply_parse()

        # Build variable vector
        my_s = np.zeros(reac_man.n_species)
        for i in range(len(reac_man.species_name_list)):
            my_s[i] = reac_man.species_density[reac_man.species_name_list[i]][0]
        my_v = np.zeros(reac_man.n_species + 1)
        my_v[0] = time_opts['T Initial']
        my_v[1:] = my_s

        # Compute function at base inputs
        RHS_T, RHS_species = reac_man.right_hand_side(0, my_v)

        # Compute zcrit portion
        reac_sys = reac_man.reaction_systems[0]
        my_rxn = reac_sys.model_list[0].my_funcs[0]
        rho_50 = my_rxn.rxn_info['BET_C6']*my_rxn.material_info['rho']*my_rxn.rxn_info['Y_Graphite']/200.0
        rho_fun = my_s[my_rxn.name_map['C6Li']]*my_s[my_rxn.name_map['EC']]/(rho_50 + my_s[my_rxn.name_map['EC']])
        crit_fun = np.exp(-my_rxn.C_t*my_rxn.z_c*my_s[my_rxn.name_map['Li2CO3']])
        conc_fun_1 = my_rxn.a_e_crit*rho_fun*crit_fun
        r_1 = -1.*my_rxn.A*conc_fun_1*np.exp(-my_rxn.EoR/time_opts['T Initial'])

        # Compute Damkohler portion with hard coded true values
        dam_info = my_rxn.rxn_info['Damkohler']
        AD = 1141791418.7518132
        EDoR = 12027.181430031871
        Da = 1.0/(1 + AD*np.exp(-EDoR/my_v[0]))

        err = np.abs(r_1[0]*Da*2*79.007/(2*79.007 + 88.062) - RHS_species[0])
        self.assertTrue(err < 1e-12, '\tFailed with RMSE {:0.2e}\n'.format(err))


if __name__ == '__main__':
    unittest.main()
