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
import unit_meshes
from unit_mocks import time_opts_mock
sys.path.append('../Source')
import conduction
import boundary
import equation_sys


class build_tests(unittest.TestCase):
    def test_cond_apply(self):
        print('\nTesting conductivity assembly...')
        # Load a simple mesh
        dx_a = 0.5
        k_a = 10.
        grid_man, mat_man = unit_meshes.mesh_one(dx_a=dx_a,k_a=k_a)
        cond_man = conduction.conduction_manager(grid_man)
        eqn_sys = equation_sys.eqn_sys(None, None, None, grid_man, False, time_opts_mock)

        # Do the apply
        cond_man.apply(eqn_sys, mat_man)

        # Calculate the expected matrix terms
        c_end = k_a/dx_a
        c_i = 2*k_a/dx_a
        s_i = -k_a/dx_a
        passing = True

        # Check error
        for i in range(grid_man.n_tot):
            if i == 0:
                err_c = (eqn_sys.LHS_c[i] - c_end)**2
                err_u = (eqn_sys.LHS_u[i] - s_i)**2
                err_l = (eqn_sys.LHS_l[i])**2
            elif i == grid_man.n_tot - 1:
                err_c = (eqn_sys.LHS_c[i] - c_end)**2
                err_u = (eqn_sys.LHS_u[i])**2
                err_l = (eqn_sys.LHS_l[i] - s_i)**2
            else:
                err_c = (eqn_sys.LHS_c[i] - c_i)**2
                err_u = (eqn_sys.LHS_u[i] - s_i)**2
                err_l = (eqn_sys.LHS_l[i] - s_i)**2
            if err_c > 10e-12:
                print('Error at center diagonal, row {}'.format(i))
                passing = False
            if err_u > 10e-12:
                print('Error at upper diagonal, row {}'.format(i))
                passing = False
            if err_l > 10e-12:
                print('Error at lower diagonal, row {}'.format(i))
                passing = False

        self.assertTrue(passing)


    def test_bc_apply(self):
        print('\nTesting boundary condition assembly...')
        # Load a simple mesh
        dx_a = 0.5
        k_a = 10.
        grid_man, mat_man = unit_meshes.mesh_one(dx_a=dx_a,k_a=k_a)
        eqn_sys = equation_sys.eqn_sys(None, None, None, grid_man, False, time_opts_mock)
        bc_man = boundary.bc_manager(grid_man)

        # Set the boundary terms
        h_ext = 10.
        T_ext = 500.
        h_left = 2.
        T_left = 101.
        h_right = 4.
        T_right = 102.
        bnd_dict = {}
        bnd_dict['External'] = {'Type': 'convection', 'h': h_ext, 'T': T_ext}
        bnd_dict['Left'] = {'Type': 'convection', 'h': h_left, 'T': T_left}
        bnd_dict['Right'] = {'Type': 'Convection', 'h': h_right, 'T': T_right}
        bc_man.setup(bnd_dict)

        # Apply boundary terms
        bc_man.apply(eqn_sys, mat_man, 0)

        phi_left = 2*k_a/dx_a
        c_left = h_left*phi_left/(h_left + phi_left)
        phi_right = 2*k_a/dx_a
        c_right = h_right*phi_right/(h_right + phi_right)

        # Check error
        h_const = h_ext*dx_a*grid_man.PA_r
        err_l = np.sum(eqn_sys.LHS_l**2)
        err_u = np.sum(eqn_sys.LHS_u**2)
        err_c = np.sum((eqn_sys.LHS_c[1:grid_man.n_tot-1] - h_const)**2)
        err_r = np.sum((eqn_sys.RHS[1:grid_man.n_tot-1] - h_const*T_ext)**2)

        err_c += (eqn_sys.LHS_c[0] - (h_const + c_left))**2
        err_c += (eqn_sys.LHS_c[grid_man.n_tot-1] - (h_const + c_right))**2

        err_r += (eqn_sys.RHS[0] - (h_const*T_ext + c_left*T_left))**2
        err_r += (eqn_sys.RHS[grid_man.n_tot-1] - (h_const*T_ext + c_right*T_right))**2

        passing = False
        err_tot = sum([err_c, err_r, err_l, err_u])
        self.assertTrue(err_tot < 10e-12, '\tFailed with error {:0.1e}\n'.format(err_tot))


if __name__ == '__main__':
    unittest.main()
