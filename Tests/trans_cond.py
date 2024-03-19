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
import scipy as sp
import time, sys, os
sys.path.append('../')
import main_fv

import matplotlib as mpl
mpl.use( 'Agg' )
import matplotlib.pyplot as plt


class trans_cond_tests(unittest.TestCase):
    def setUp(self):
        self.plotting = True


    def test_trans_end_conv(self):
        print('\nTesting transient symmetric convection...')
        # Supply file name
        file_name = os.getcwd() + '/Inputs/trans_end_conv.yaml'

        # Run model
        model = main_fv.lim1tr_model(file_name)
        eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()
        T_sol = data_man.data_dict['Temperature'][-1,:]

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
        err = np.sqrt(np.sum((T_ans - T_sol[half_nodes:])**2)/half_nodes)
        if self.plotting:
            plt.figure()
            plt.plot(L*(1. + x_star), T_ans, 'o', label='Analytical')
            plt.plot(grid_man.x_node, T_sol, '-', label='Numerical')
            plt.ylim([370, 470])
            plt.xlabel(r'Postion ($m$)')
            plt.ylabel(r'Temperature ($K$)')
            plt.legend()
            plt.title('RMSE = {:.2E}'.format(err))
            plt.savefig('./Figures/trans_end_conv.png', bbox_inches='tight')
            plt.close()

        self.assertTrue(err < 2e-2,'\tFailed with RMSE {:0.2e}\n'.format(err))


    def test_trans_ext_conv(self):
        print('\nTesting transient external convection...')
        # Supply file name
        file_name = os.getcwd() + '/Inputs/trans_ext_conv.yaml'

        # Run model
        model = main_fv.lim1tr_model(file_name)
        eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()
        T_sol = data_man.data_dict['Temperature'][-1,:]

        my_t = time_opts['Run Time']
        my_mat = mat_man.get_material('A')
        h_ext = model.parser.cap_dict['Boundary']['External']['h']
        T_ext = model.parser.cap_dict['Boundary']['External']['T']
        C_o = h_ext*bc_man.PA_r/(my_mat.rho*my_mat.cp)
        T_ans = T_ext + (np.mean(time_opts['T Initial']) - T_ext)*np.exp(-1.0*C_o*my_t)
        err = np.max(np.abs(T_sol - T_ans))
        self.assertTrue(err < 3e-2, '\tFailed with RMSE {:0.2e}\n'.format(err))


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
        model.parser.cap_dict['Materials']['A']['k'] = 500.
        model.parser.cap_dict['Time']['dt'] = 0.01
        eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()
        T_sol = data_man.data_dict['Temperature'][-1,:]

        dT_rate = 2*10000/(0.01*2000*500)
        T_true = 300 + dT_rate*5

        err = abs(T_true - T_sol[0])
        self.assertTrue(err < 5e-2, '\tFailed with RMSE {:0.2e}\n'.format(err))


    def test_ramp_controlled_bc(self):
        print('\nTesting heating rate boundary deactivation...')
        # Build model
        file_name = os.getcwd() + '/Inputs/small_cube.yaml'
        model = main_fv.lim1tr_model(file_name)

        # Set temperature control on left BC
        control_bc = {'Type': 'Dirichlet',
                      'T': {'Initial': 300, 'Rate': 5},
                      'Temperature Control': {
                        'T Location': 0,
                        'T Cutoff': 325,
                        'T Post': 300.,
                        'h Post': 0}
                     }

        bnd_dict = {'External': {'Type': 'Adiabatic'},
            'Left': control_bc,
            'Right': {'Type': 'Adiabatic'}}
        model.parser.cap_dict['Boundary'] = bnd_dict
        model.parser.cap_dict['Materials']['A']['k'] = 500.
        eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()
        T_true = 325.125
        T_sol = data_man.data_dict['Temperature'][-1,:]

        err = abs(T_true - T_sol[0])
        self.assertTrue(err < 1e-8, '\tFailed with RMSE {:0.2e}\n'.format(err))


    def test_flux_controlled_bc(self):
        print('\nTesting flux boundary deactivation...')
        # Build model
        file_name = os.getcwd() + '/Inputs/small_cube.yaml'
        model = main_fv.lim1tr_model(file_name)

        # Set flux control on left BC
        control_bc = {'Type': 'Heat Flux',
                      'Flux': 50000.,
                      'Temperature Control': {
                        'T Location': 0,
                        'T Cutoff': 325,
                        'T Post': 300.,
                        'h Post': 0}
                     }
        bnd_dict = {'External': {'Type': 'Adiabatic'},
            'Left': control_bc,
            'Right': {'Type': 'Adiabatic'}}
        model.parser.cap_dict['Boundary'] = bnd_dict
        model.parser.cap_dict['Materials']['A']['k'] = 500.
        eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()
        T_true = 325.0
        T_sol = data_man.data_dict['Temperature'][-1,:]

        err = abs(T_true - T_sol[0])
        self.assertTrue(err < 1e-8, '\tFailed with RMSE {:0.2e}\n'.format(err))


if __name__ == '__main__':
    unittest.main()
