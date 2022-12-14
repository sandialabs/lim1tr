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
import boundary_types

import matplotlib as mpl
mpl.use( 'Agg' )
import matplotlib.pyplot as plt


class rad_tests(unittest.TestCase):
    def setUp(self):
        self.plotting = True


    def test_steady_rad(self):
        print('\nTesting steady radiation...')
        file_name = os.getcwd() + '/Inputs/rad_bc.yaml'

        # Run model
        model = main_fv.lim1tr_model(file_name)
        eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()

        # Analytical solution
        cap_dict = model.parser.cap_dict
        T_l_4 = cap_dict['Boundary']['Left']['T']**4
        T_r_4 = cap_dict['Boundary']['Right']['T']**4
        eps_l = cap_dict['Boundary']['Left']['eps']
        eps_r = cap_dict['Boundary']['Right']['eps']
        k = cap_dict['Materials']['A']['k']
        L = cap_dict['Domain Table']['Thickness'][0]
        sig = 5.67e-8
        kol = k/L
        es_l = eps_l*sig
        es_r = eps_r*sig

        F = np.zeros(2)
        x = np.array([500., 300.])
        J = np.zeros([2,2])
        J[0,1] = kol
        J[1,0] = -1*kol
        for i in range(15):
            J[0,0] = -4*es_l*x[0]**3 - kol
            J[1,1] = 4*es_r*x[1]**3 + kol
            F[0] = es_l*(T_l_4 - x[0]**4) - kol*(x[0] - x[1])
            F[1] = es_r*(x[1]**4 - T_r_4) - kol*(x[0] - x[1])
            dx = np.dot(np.linalg.inv(J), -1*F)
            x += dx
            if (max(abs(dx)) < 1e-12):
                break
        x_arr = np.array([0, L])
        T_arr = np.array([x[0], x[1]])
        T_ans = np.interp(grid_man.x_node, x_arr, T_arr)

        err = np.sum((T_ans - eqn_sys.T_sol)**2)/grid_man.n_tot

        if self.plotting:
            raw_diff = False
            plt.figure()
            if raw_diff:
                plt.plot(grid_man.x_node, T_ans - eqn_sys.T_sol, '-', label='Analytical - Numerical')
            else:
                plt.plot(grid_man.x_node, T_ans, 'o', label='Analytical')
                plt.plot(grid_man.x_node, eqn_sys.T_sol, '-', label='Numerical')
            plt.xlabel('Postion (m)')
            plt.ylabel('Temperature (K)')
            plt.legend()
            plt.title('MSE = {:.3E}'.format(err))
            plt.savefig('./Figures/steady_rad.png', bbox_inches='tight')
            plt.close()


    def lumped_rad(self, T, dt, phi, T_i, T_e):
        T_e_2 = T_e**2
        T_e_3 = T_e**3

        g_n = T*(T_i + T_e) - T_i*T_e - T_e_2
        g_d = T*(T_i - T_e) + T_i*T_e - T_e_2

        g_n_p = T_i + T_e
        g_d_p = T_i - T_e

        g = g_n/g_d
        g_p = (g_n_p*g_d - g_d_p*g_n)/g_d**2

        my_f = (-0.25*phi/T_e_3)*(np.log(g) - 2*(np.arctan(T/T_e) - np.arctan(T_i/T_e))) - dt
        my_j = (-0.25*phi/T_e_3)*((g_p/g) - 2*T_e/(T**2 + T_e_2))

        return my_f, my_j


    def test_trans_rad(self):
        print('\nTesting transient radiation with first order time...')
        file_name = os.getcwd() + '/Inputs/trans_rad_bc.yaml'

        # Run model
        model = main_fv.lim1tr_model(file_name)
        eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()

        t_sim = data_man.data_dict['Time'].flatten()
        T_sim = data_man.data_dict['Temperature'].flatten()

        # Analytical solution
        cap_dict = model.parser.cap_dict
        sig = 5.67e-8
        L_x = cap_dict['Domain Table']['Thickness'][0]
        L_y = cap_dict['Other']['Y Dimension']
        L_z = cap_dict['Other']['Z Dimension']
        A_c = L_y*L_z
        V = A_c*L_x
        A_rad = 2*A_c + 2*L_x*(L_y + L_z)
        phi = V*cap_dict['Materials']['A']['rho']*cap_dict['Materials']['A']['cp']
        phi = phi/(cap_dict['Boundary']['Left']['eps']*sig*A_rad)
        T_i = cap_dict['Time']['T Initial'][0]
        T_e = cap_dict['Boundary']['Left']['T']
        dt = cap_dict['Time']['dt']

        T_arr = np.zeros(t_sim.shape[0])
        T_arr[0] = 1*T_i
        for i in range(1,t_sim.shape[0]):
            dt = t_sim[i]
            dT = 1
            T = 1*T_i
            j = 0
            while (np.abs(dT) > 1e-13) and (j < 30):
                my_f, my_j = self.lumped_rad(T, dt, phi, T_i, T_e)
                dT = -my_f/my_j
                T += dT
                j += 1
            T_arr[i] = 1*T

        err = np.sum((T_arr - T_sim)**2)/t_sim.shape[0]

        if self.plotting:
            raw_diff = False
            plt.figure()
            if raw_diff:
                plt.plot(t_sim, T_arr - T_sim, '-', label='Analytical - Numerical')
            else:
                plt.plot(t_sim, T_arr, 'o', label='Analytical')
                plt.plot(t_sim, T_sim, '-', label='Numerical')
            plt.xlabel('Time (s)')
            plt.ylabel('Temperature (K)')
            plt.legend()
            plt.title('MSE = {:.3E}'.format(err))
            plt.savefig('./Figures/trans_rad.png', bbox_inches='tight')
            plt.close()

        self.assertTrue(err < 2e-6, '\tFailed with MSE {:0.2e}\n'.format(err))


    def test_arc_bc(self):
        print('\nTesting arc radiation bc...')
        bc = boundary_types.end_radiation_arc(np.ones(1), 'Right')
        bc.set_params(0.5, 300, 1)
        dt = 5.
        T_node = np.array([[300], [300], [301], [307], [308]])
        T_arc = np.zeros(T_node.shape[0])
        for n in range(T_node.shape[0]):
            bc.update_params(T_node[n,:], dt)
            bc.update_post_step()
            T_arc[n] = bc.T_ext
        T_arc_true = np.array([300., 300., 301., 306., 308.])
        err = np.sum(np.abs(T_arc-T_arc_true))
        self.assertTrue(err < 1e-15, '\tFailed total error {:0.2e}\n'.format(err))


if __name__ == '__main__':
    unittest.main()
