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
import scipy as sp
import time, sys, os
sys.path.append('../')
import main_fv

import matplotlib as mpl
mpl.use( 'Agg' )
import matplotlib.pyplot as plt


class trans_cond_tests(unittest.TestCase):
    def setUp(self):
        self.plotting = False


    def test_trans_end_conv_bdf1(self):
        print('\nTesting first-order transient symmetric convection...')
        # Supply file name
        file_name = os.getcwd() + '/Inputs/trans_end_conv_bdf1.yaml'
        err = self.trans_end_conv(file_name)
        self.assertTrue(err < 2e-2,'\tFailed with RMSE {:0.2e}\n'.format(err))


    def test_trans_end_conv_cn(self):
        print('\nTesting second-order transient symmetric convection...')
        # Supply file name
        file_name = os.getcwd() + '/Inputs/trans_end_conv_cn.yaml'
        err = self.trans_end_conv(file_name)
        self.assertTrue(err < 2e-2,'\tFailed with RMSE {:0.2e}\n'.format(err))


    def test_trans_end_conv_bdf1_split(self):
        print('\nTesting first-order split step transient symmetric convection...')
        # Supply file name
        file_name = os.getcwd() + '/Inputs/trans_end_conv_bdf1_split.yaml'
        err = self.trans_end_conv(file_name)
        self.assertTrue(err < 2e-2,'\tFailed with RMSE {:0.2e}\n'.format(err))


    def test_trans_end_conv_cn_split(self):
        print('\nTesting second-order split step transient symmetric convection...')
        # Supply file name
        file_name = os.getcwd() + '/Inputs/trans_end_conv_cn_split.yaml'
        err = self.trans_end_conv(file_name)
        self.assertTrue(err < 2e-2,'\tFailed with RMSE {:0.2e}\n'.format(err))


    def trans_end_conv(self, file_name):
        # Run model
        model = main_fv.lim1tr_model(file_name)
        eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()

        # Fourier number
        L = np.sum(grid_man.dx_arr)*0.5
        my_t = time_opts['Run Time']
        my_mat = mat_man.get_material('A')
        alpha = my_mat.k/(my_mat.rho*my_mat.cp)
        Fo = alpha*my_t/L**2

        # Analytical soln (Incropera 6th edition, p. 273)
        zeta_n = [1.3138, 4.0336, 6.9096, 9.8928] # First four roots of the transcendental eqn with Bi = 5
        half_nodes = int(grid_man.n_tot*0.5)
        x_star = (np.arange(half_nodes) + 0.5)/half_nodes
        T_right = model.parser.cap_dict['Boundary']['Right']['T']
        theta = np.zeros(half_nodes)
        for i in range(4):
            C_n = 4.*np.sin(zeta_n[i])/(2.*zeta_n[i] + np.sin(2.*zeta_n[i]))
            theta += C_n*np.exp(-zeta_n[i]**2*Fo)*np.cos(zeta_n[i]*x_star)
        T_ans = T_right + theta*(np.mean(time_opts['T Initial']) - T_right)

        # Calculate error
        err = np.sqrt(np.sum((T_ans - eqn_sys.T_sol[half_nodes:])**2)/half_nodes)
        if self.plotting:
            is_split = ''
            if 'split' in file_name:
                is_split = '_split'
            plt.figure()
            plt.plot(L*(1. + x_star), T_ans, 'o', label='Analytical')
            plt.plot(grid_man.x_node, eqn_sys.T_sol, '-', label='Numerical')
            plt.ylim([370, 470])
            plt.xlabel(r'Postion ($m$)')
            plt.ylabel(r'Temperature ($K$)')
            plt.legend()
            plt.title('RMSE = {:.2E}'.format(err))
            plt.savefig('./Figures/trans_end_conv_order_{}{}.png'.format(time_opts['Order'], is_split), bbox_inches='tight')
            plt.close()

        return err


    def test_trans_ext_conv_bdf1(self):
        print('\nTesting first-order transient external convection...')
        # Supply file name
        file_name = os.getcwd() + '/Inputs/trans_ext_conv_bdf1.yaml'
        err = self.trans_ext_conv(file_name)
        self.assertTrue(err < 3e-2, '\tFailed with RMSE {:0.2e}\n'.format(err))


    def test_trans_ext_conv_cn(self):
        print('\nTesting first-order transient external convection...')
        # Supply file name
        file_name = os.getcwd() + '/Inputs/trans_ext_conv_cn.yaml'
        err = self.trans_ext_conv(file_name)
        self.assertTrue(err < 3.5e-6, '\tFailed with RMSE {:0.2e}\n'.format(err))


    def test_trans_ext_conv_bdf1_split(self):
        print('\nTesting first-order transient external convection...')
        # Supply file name
        file_name = os.getcwd() + '/Inputs/trans_ext_conv_bdf1_split.yaml'
        err = self.trans_ext_conv(file_name)
        self.assertTrue(err < 1.5e-2, '\tFailed with RMSE {:0.2e}\n'.format(err))


    def test_trans_ext_conv_cn_split(self):
        print('\nTesting first-order transient external convection...')
        # Supply file name
        file_name = os.getcwd() + '/Inputs/trans_ext_conv_cn_split.yaml'
        err = self.trans_ext_conv(file_name)
        self.assertTrue(err < 8e-7, '\tFailed with RMSE {:0.2e}\n'.format(err))


    def trans_ext_conv(self, file_name):
        # Run model
        model = main_fv.lim1tr_model(file_name)
        eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()

        my_t = time_opts['Run Time']
        my_mat = mat_man.get_material('A')
        h_ext = model.parser.cap_dict['Boundary']['External']['h']
        T_ext = model.parser.cap_dict['Boundary']['External']['T']
        C_o = h_ext*bc_man.PA_r/(my_mat.rho*my_mat.cp)
        T_ans = T_ext + (np.mean(time_opts['T Initial']) - T_ext)*np.exp(-1.0*C_o*my_t)
        err = np.max(np.abs(eqn_sys.T_sol - T_ans))
        return err


    def test_trans_end_flux_cn(self):
        print('\nTesting second-order transient end flux...')
        # Run model
        file_name = os.getcwd() + '/Inputs/trans_end_flux_cn.yaml'
        model = main_fv.lim1tr_model(file_name)
        eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()

        # Save a few numbers
        L = np.sum(grid_man.dx_arr)*0.5
        my_t = time_opts['Run Time']
        my_mat = mat_man.get_material('A')
        alpha = my_mat.k/(my_mat.rho*my_mat.cp)
        q_in = model.parser.cap_dict['Boundary']['Left']['Flux']

        # Analytical soln (Incropera 6th edition, p. 286)
        c_one = (2*q_in/my_mat.k)*np.sqrt(alpha*my_t/np.pi)
        c_two = np.exp(-1.*grid_man.x_node**2/(4*alpha*my_t))
        c_three = q_in*grid_man.x_node/my_mat.k
        c_four = sp.special.erfc(grid_man.x_node*0.5/np.sqrt(alpha*my_t))
        T_ans = np.mean(time_opts['T Initial']) + c_one*c_two - c_three*c_four

        # Calculate error
        err = np.sqrt(np.sum((T_ans - eqn_sys.T_sol)**2)/grid_man.n_tot)
        if self.plotting:
            is_split = ''
            if 'split' in file_name:
                is_split = '_split'
            plt.figure()
            plt.plot(grid_man.x_node, T_ans, 'o', label='Analytical')
            plt.plot(grid_man.x_node, eqn_sys.T_sol, '-', label='Numerical')
            plt.xlabel(r'Postion ($m$)')
            plt.ylabel(r'Temperature ($K$)')
            plt.legend()
            plt.title('RMSE = {:.2E}'.format(err))
            plt.savefig('./Figures/trans_end_flux_cn.png', bbox_inches='tight')
            plt.close()

        self.assertTrue(err < 2e-4, '\tFailed with RMSE {:0.2e}\n'.format(err))


    def test_deactivate_bcs(self):
        print('\nTesting flux boundary deactivation...')
        # Build model
        file_name = os.getcwd() + '/Inputs/small_cube.yaml'
        model = main_fv.lim1tr_model(file_name)

        # Set steady flux on all BCs
        flux_bnd = {'Type': 'Heat Flux',
            'Flux': 10000.,
            'Deactivation Time': 5}
        bnd_dict = {'External': {'Type': 'Adiabatic'},
            'Left': flux_bnd,
            'Right': flux_bnd}
        model.parser.cap_dict['Boundary'] = bnd_dict
        eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()

        dT_rate = 2*10000/(0.01*2000*500)
        T_true = 300 + dT_rate*5

        err = abs(T_true - eqn_sys.T_sol[0])
        self.assertTrue(err < 1e-13, '\tFailed with RMSE {:0.2e}\n'.format(err))


    def test_controlled_bc(self):
        print('\nTesting flux boundary deactivation...')
        # Build model
        file_name = os.getcwd() + '/Inputs/small_cube.yaml'
        model = main_fv.lim1tr_model(file_name)

        # Set temperature control on left BC
        control_bc = {'Type': 'Temperature Control',
            'T': 300.,
            'T Rate': 5,
            'T Cutoff': 325,
            'T End': 300.,
            'h': 0}
        bnd_dict = {'External': {'Type': 'Adiabatic'},
            'Left': control_bc,
            'Right': {'Type': 'Adiabatic'}}
        model.parser.cap_dict['Boundary'] = bnd_dict
        model.parser.cap_dict['Materials']['A']['k'] = 50.
        eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()
        T_true = 321.01094462

        err = abs(T_true - eqn_sys.T_sol[0])
        self.assertTrue(err < 1e-8, '\tFailed with RMSE {:0.2e}\n'.format(err))


if __name__ == '__main__':
    unittest.main()
