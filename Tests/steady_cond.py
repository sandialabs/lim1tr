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
import numpy as np
import time, sys, os
sys.path.append('../')
import main_fv

import matplotlib as mpl
mpl.use( 'Agg' )
import matplotlib.pyplot as plt
# latex options
plt.rc( 'text', usetex = True )
plt.rc( 'font', family = 'serif' )
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


def quick_plot(x_node, T_ans, T_lin, MSE, fig_name, raw_diff=False):
    plt.figure()
    if raw_diff:
        plt.plot(x_node, T_ans - T_lin, '-', label='Analytical - Numerical')
    else:
        plt.plot(x_node, T_ans, 'o', label='Analytical')
        plt.plot(x_node, T_lin, '-', label='Numerical')
    plt.xlabel(r'Postion ($m$)')
    plt.ylabel(r'Temperature ($K$)')
    plt.legend()
    plt.title('MSE = {:.3E}'.format(MSE))
    plt.savefig('./Figures/' + fig_name + '.png', bbox_inches='tight')
    plt.close()


def simple_steady_cond(plotting=False):
    print('Testing steady dirichlet conduction...')
    # Supply file name
    file_name = os.getcwd() + '/Inputs/simple_cond.yaml'

    # Run model
    model = main_fv.lim1tr_model(file_name)
    eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, time_opts = model.run_model()

    # Answer should be a line
    Lx = np.sum(grid_man.dx_arr)
    r_tot = (1./bc_man.h_left) + (Lx/mat_man.k_arr[0]) + (1./bc_man.h_right)
    q_c = (bc_man.T_left - bc_man.T_right)/r_tot
    T_1 = bc_man.T_left - q_c/bc_man.h_left
    T_2 = bc_man.T_right + q_c/bc_man.h_right
    x_arr = np.array([0, Lx])
    T_arr = np.array([T_1, T_2])
    T_ans = np.interp(grid_man.x_node, x_arr, T_arr)

    err = np.sum((T_ans - eqn_sys.T_lin)**2)/grid_man.n_tot
    if plotting:
        quick_plot(grid_man.x_node, T_ans, eqn_sys.T_lin, err, 'simple_steady_cond')

    if err > 1e-16:
        print('\tFailed with MSE {:0.2e}\n'.format(err))
        return 0
    else:
        print('\tPassed\n')
        return 1


def end_conv_steady_cond(plotting=False):
    print('Testing steady conduction with convection on ends...')
    # Supply file name
    file_name = os.getcwd() + '/Inputs/simple_conv.yaml'

    # Run model
    model = main_fv.lim1tr_model(file_name)
    eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, time_opts = model.run_model()

    # Analytical soln
    r_tot = (2./100.) + np.sum(grid_man.dx_arr)/mat_man.k_arr[0]
    q_flux = (bc_man.T_left - bc_man.T_right)/r_tot
    T_o = bc_man.T_left - q_flux/100.
    T_l = bc_man.T_right + q_flux/100.
    T_ans = T_o - (q_flux/mat_man.k_arr[0])*grid_man.x_node

    err = np.sum((T_ans - eqn_sys.T_lin)**2)/grid_man.n_tot
    if plotting:
        quick_plot(grid_man.x_node, T_ans, eqn_sys.T_lin, err, 'end_conv_steady_cond')

    if err > 1e-16:
        print('\tFailed with MSE {:0.2e}\n'.format(err))
        return 0
    else:
        print('\tPassed\n')
        return 1


def end_conv_steady_cond_stack(plotting=False):
    print('Testing steady conduction through a stack of three materials...')
    # Supply file name
    file_name = os.getcwd() + '/Inputs/stack_cond.yaml'

    # Run model
    model = main_fv.lim1tr_model(file_name)
    eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, time_opts = model.run_model()

    # Analytical soln
    r_tot = 2*(0.1/10.) + 0.1/2. + 2./10000.
    q_flux = (bc_man.T_left - bc_man.T_right)/r_tot
    T_b = np.zeros(4)
    T_b[0] = bc_man.T_left - q_flux/bc_man.h_left
    T_b[1] = T_b[0] - q_flux*0.1/10.
    T_b[2] = T_b[1] - q_flux*0.1/2.
    T_b[3] = T_b[2] - q_flux*0.1/10.
    x_b = np.array([0, 0.1, 0.2, 0.3])
    T_ans = np.interp(grid_man.x_node, x_b, T_b)

    err = np.sum((T_ans - eqn_sys.T_lin)**2)/grid_man.n_tot
    if plotting:
        quick_plot(grid_man.x_node, T_ans, eqn_sys.T_lin, err, 'end_conv_steady_cond_stack')

    if err > 1e-16:
        print('\tFailed with MSE {:0.2e}\n'.format(err))
        return 0
    else:
        print('\tPassed\n')
        return 1


def exterior_steady_cond(plotting=False):
    print('Testing steady dirichlet conduction with a convection source at each node...')
    # Supply file name
    file_name = os.getcwd() + '/Inputs/simple_fin.yaml'

    # Run model
    model = main_fv.lim1tr_model(file_name)
    eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, time_opts = model.run_model()

    # Analytical soln (Incropera 6th edition, p. 144)
    dy = 0.2
    dz = 0.1
    P = 2*(dy + dz)
    A_c = dy*dz
    L_x = np.sum(grid_man.dx_arr)
    C_m = np.sqrt(bc_man.h_ext*bc_man.PA_r/mat_man.k_arr[0])
    s_grid = grid_man.x_node
    C_1 = 1./((1. + np.exp(C_m*L_x)) - (mat_man.k_arr[0]*C_m/bc_man.h_left)*(1. - np.exp(C_m*L_x)))
    T_ans = bc_man.T_ext + (bc_man.T_left - bc_man.T_ext)*C_1*(np.exp(C_m*s_grid) + np.exp(C_m*(L_x - s_grid)))

    err = np.sum((T_ans - eqn_sys.T_lin)**2)/grid_man.n_tot
    if plotting:
        quick_plot(grid_man.x_node, T_ans, eqn_sys.T_lin, err, 'exterior_steady_cond')

    if err > 2e-5:
        print('\tFailed with MSE {:0.2e}\n'.format(err))
        return 0
    else:
        print('\tPassed\n')
        return 1


def contact_resistance(plotting=False):
    print('Testing steady conduction with through a stack of three materials with contact resistances...')
    # Supply file name
    file_name = os.getcwd() + '/Inputs/stack_contact.yaml'

    # Run model
    model = main_fv.lim1tr_model(file_name)
    eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, time_opts = model.run_model()

    # Analytical soln
    R_1 = 1./100.
    R_2 = 2./100.
    mat_nodes = int(grid_man.x_node.shape[0]/3.)
    r_tot = 2*(0.1/10.) + 0.1/2. + 1./bc_man.h_left + 1./bc_man.h_right + R_1 + R_2
    q_flux = (bc_man.T_left - bc_man.T_right)/r_tot
    T_1 = bc_man.T_left - q_flux/bc_man.h_left
    T_2 = T_1 - q_flux*0.1/10.
    T_3 = T_2 - q_flux*R_1
    T_4 = T_3 - q_flux*0.1/2.
    T_5 = T_4 - q_flux*R_2
    T_6 = T_5 - q_flux*0.1/10.
    T_ans = np.zeros(grid_man.x_node.shape[0])
    T_ans[:mat_nodes] = grid_man.x_node[:mat_nodes]*(T_2 - T_1)/0.1 + T_1
    T_ans[mat_nodes:2*mat_nodes] = (grid_man.x_node[mat_nodes:2*mat_nodes] - 0.1)*(T_4 - T_3)/0.1 + T_3
    T_ans[2*mat_nodes:] = (grid_man.x_node[2*mat_nodes:] - 0.2)*(T_6 - T_5)/0.1 + T_5

    err = np.sum((T_ans - eqn_sys.T_lin)**2)/grid_man.n_tot
    if plotting:
        quick_plot(grid_man.x_node, T_ans, eqn_sys.T_lin, err, 'end_conv_steady_contact_stack', raw_diff=False)

    if err > 1e-1:
        print('\tFailed with MSE {:0.2e}\n'.format(err))
        return 0
    else:
        print('\tPassed\n')
        return 1


def left_flux_right_conv(plotting=False):
    print('Testing steady conduction with through a stack of three materials with contact resistances...')
    # Supply file name
    file_name = os.getcwd() + '/Inputs/left_flux_right_conv.yaml'

    # Run model
    model = main_fv.lim1tr_model(file_name)
    eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, time_opts = model.run_model()

    # Analytical soln
    L_x = np.sum(grid_man.dx_arr)
    T_r = bc_man.T_right + bc_man.flux_left/bc_man.h_right
    T_l = T_r + bc_man.flux_left*L_x/mat_man.k_arr[0]
    T_ans = T_l + grid_man.x_node*(T_r - T_l)/L_x

    err = np.sum((T_ans - eqn_sys.T_lin)**2)/grid_man.n_tot
    if plotting:
        quick_plot(grid_man.x_node, T_ans, eqn_sys.T_lin, err, 'left_flux_right_conv', raw_diff=False)

    if err > 1e-12:
        print('\tFailed with MSE {:0.2e}\n'.format(err))
        return 0
    else:
        print('\tPassed\n')
        return 1


def left_conv_right_flux(plotting=False):
    print('Testing steady conduction with through a stack of three materials with contact resistances...')
    # Supply file name
    file_name = os.getcwd() + '/Inputs/left_conv_right_flux.yaml'

    # Run model
    model = main_fv.lim1tr_model(file_name)
    eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, time_opts = model.run_model()

    # Analytical soln
    L_x = np.sum(grid_man.dx_arr)
    T_l = bc_man.T_left + bc_man.flux_right/bc_man.h_left
    T_r = T_l + bc_man.flux_right*L_x/mat_man.k_arr[0]
    T_ans = T_l + grid_man.x_node*(T_r - T_l)/L_x

    err = np.sum((T_ans - eqn_sys.T_lin)**2)/grid_man.n_tot
    if plotting:
        quick_plot(grid_man.x_node, T_ans, eqn_sys.T_lin, err, 'left_conv_right_flux', raw_diff=False)

    if err > 1e-12:
        print('\tFailed with MSE {:0.2e}\n'.format(err))
        return 0
    else:
        print('\tPassed\n')
        return 1


if __name__ == '__main__':
    simple_steady_cond(plotting=True)

    end_conv_steady_cond(plotting=True)

    end_conv_steady_cond_stack(plotting=True)

    exterior_steady_cond(plotting=True)

    contact_resistance(plotting=True)

    left_flux_right_conv(plotting=True)

    left_conv_right_flux(plotting=True)
