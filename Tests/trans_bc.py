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
from scipy.integrate import odeint
import sys, os
sys.path.append('../')
import main_fv

import matplotlib as mpl
mpl.use( 'Agg' )
import matplotlib.pyplot as plt


class trans_bc_tests(unittest.TestCase):
    def setUp(self):
        self.plotting = True


    def test_lumped_t_inf_ramp_side(self):
        print('\nTesting time dependent convection temperature ramp on the sides...')
        T_ramp = {'Initial': 500, 'Rate': 4}
        err = self.trans_temperature(T_ramp, 'lumped_t_inf_side_ramp')
        self.assertTrue(err < 1e-2,'\tFailed with RMSE {:0.2e}\n'.format(err))


    def test_lumped_t_inf_table_side(self):
        print('\nTesting time dependent convection temperature table on the sides...')
        T_ramp = {'Table': './Inputs/t_ramp.csv'}
        err = self.trans_temperature(T_ramp, 'lumped_t_inf_side_table')
        self.assertTrue(err < 1e-2,'\tFailed with RMSE {:0.2e}\n'.format(err))


    def trans_temperature(self, T_ramp, fig_name):
        # Supply file name
        file_name = os.getcwd() + '/Inputs/lumped.yaml'

        # Run model
        model = main_fv.lim1tr_model(file_name)
        model.parser.cap_dict['Boundary']['Left']['T'] = T_ramp
        model.parser.cap_dict['Boundary']['Right']['T'] = T_ramp
        eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()
        t_sol = data_man.data_dict['Time']
        T_sol = data_man.data_dict['Temperature']

        # Analytical solution
        h_inf = model.parser.cap_dict['Boundary']['Left']['h']
        T_inf = 500
        A_s = 2*0.01**2
        V = 0.01**3
        mass =  model.parser.cap_dict['Materials']['A']['rho']*V
        c_p = model.parser.cap_dict['Materials']['A']['cp']
        T_rate = 4
        C_o = T_rate*mass*c_p/(h_inf*A_s*(T_sol[0,0] - T_inf))
        t_star = h_inf*A_s*t_sol/(mass*c_p)
        T_star = (1 + C_o)*np.exp(-t_star) + C_o*(t_star - 1)
        T_ans = (T_sol[0,0] - T_inf)*T_star + T_inf

        # Calculate error
        err = np.sqrt(np.sum((T_ans - T_sol[:,0])**2)/t_sol.shape[0])
        if self.plotting:
            plt.figure()
            plt.plot(t_sol, T_ans, 'o', label='Analytical')
            plt.plot(t_sol, T_sol[:,0], '-', label='Numerical')
            plt.plot(t_sol, T_sol[:,1], '--')
            plt.xlabel(r'Time (s)')
            plt.ylabel(r'Temperature (K)')
            plt.legend()
            plt.title('RMSE = {:.2E}'.format(err))
            plt.savefig(f'./Figures/{fig_name}.png', bbox_inches='tight')
            plt.close()

        return err


    def test_lumped_h_ramp_side(self):
        print('\nTesting time dependent convection temperature table on the sides...')
        h_ramp = {'Initial': 10, 'Rate': 2}
        # Supply file name
        file_name = os.getcwd() + '/Inputs/lumped.yaml'

        # Run model
        model = main_fv.lim1tr_model(file_name)
        model.parser.cap_dict['Boundary']['Left']['h'] = h_ramp
        model.parser.cap_dict['Boundary']['Right']['h'] = h_ramp
        eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()
        t_sol = data_man.data_dict['Time']
        T_sol = data_man.data_dict['Temperature']

        # Analytical solution
        h_o = 10
        h_rate = 2
        T_inf = 500
        A_s = 2*0.01**2
        V = 0.01**3
        mass =  model.parser.cap_dict['Materials']['A']['rho']*V
        c_p = model.parser.cap_dict['Materials']['A']['cp']
        C_o = h_rate*mass*c_p/(h_o*h_o*A_s)

        def my_ode(T, t):
            return -T*(1 + C_o*t)

        t_star = h_o*A_s*t_sol/(mass*c_p)
        T_star = odeint(my_ode, 1.0, t_star)
        T_ans = (T_sol[0,0] - T_inf)*T_star.flatten() + T_inf

        # Calculate error
        err = np.sqrt(np.sum((T_ans - T_sol[:,0])**2)/t_sol.shape[0])
        if self.plotting:
            plt.figure()
            plt.plot(t_sol, T_ans, 'o', label='ODE')
            plt.plot(t_sol, T_sol[:,0], '-', label='Numerical')
            plt.plot(t_sol, T_sol[:,1], '--')
            plt.xlabel(r'Time (s)')
            plt.ylabel(r'Temperature (K)')
            plt.legend()
            plt.title('RMSE = {:.2E}'.format(err))
            plt.savefig(f'./Figures/lumped_h_side_ramp.png', bbox_inches='tight')
            plt.close()

        self.assertTrue(err < 1e-2,'\tFailed with RMSE {:0.2e}\n'.format(err))

if __name__ == '__main__':
    unittest.main()
