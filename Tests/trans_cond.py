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
import scipy as sp
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


def trans_end_conv_bdf1(plotting=False):
    print('Testing first-order transient symmetric convection...')
    # Supply file name
    file_name = os.getcwd() + '/Inputs/trans_end_conv_bdf1.yaml'
    return trans_end_conv(file_name, plotting=plotting)


def trans_end_conv_cn(plotting=False):
    print('Testing second-order transient symmetric convection...')
    # Supply file name
    file_name = os.getcwd() + '/Inputs/trans_end_conv_cn.yaml'
    return trans_end_conv(file_name, plotting=plotting)


def trans_end_conv_bdf1_split(plotting=False):
    print('Testing first-order split step transient symmetric convection...')
    # Supply file name
    file_name = os.getcwd() + '/Inputs/trans_end_conv_bdf1_split.yaml'
    return trans_end_conv(file_name, plotting=plotting)


def trans_end_conv_cn_split(plotting=False):
    print('Testing second-order split step transient symmetric convection...')
    # Supply file name
    file_name = os.getcwd() + '/Inputs/trans_end_conv_cn_split.yaml'
    return trans_end_conv(file_name, plotting=plotting)


def trans_end_conv(file_name, plotting=False):
    # Run model
    model = main_fv.lim1tr_model(file_name)
    eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, time_opts = model.run_model()

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
    theta = np.zeros(half_nodes)
    for i in range(4):
        C_n = 4.*np.sin(zeta_n[i])/(2.*zeta_n[i] + np.sin(2.*zeta_n[i]))
        theta += C_n*np.exp(-zeta_n[i]**2*Fo)*np.cos(zeta_n[i]*x_star)
    T_ans = bc_man.T_right + theta*(time_opts['T Initial'] - bc_man.T_right)

    # Calculate error
    err = np.sqrt(np.sum((T_ans - eqn_sys.T_lin[half_nodes:])**2)/half_nodes)
    if plotting:
        is_split = ''
        if 'split' in file_name:
            is_split = '_split'
        plt.figure()
        plt.plot(L*(1. + x_star), T_ans, 'o', label='Analytical')
        plt.plot(grid_man.x_node, eqn_sys.T_lin, '-', label='Numerical')
        plt.ylim([370, 470])
        plt.xlabel(r'Postion ($m$)')
        plt.ylabel(r'Temperature ($K$)')
        plt.legend()
        plt.title('RMSE = {:.2E}'.format(err))
        plt.savefig('./Figures/trans_end_conv_order_{}{}.png'.format(time_opts['Order'], is_split), bbox_inches='tight')
        plt.close()

    if err > 2e-2:
        print('\tFailed with RMSE {:0.2e}\n'.format(err))
        return 0
    else:
        print('\tPassed\n')
        return 1


def trans_ext_conv_bdf1():
    print('Testing first-order transient external convection...')
    # Supply file name
    file_name = os.getcwd() + '/Inputs/trans_ext_conv_bdf1.yaml'
    return trans_ext_conv(file_name, 3e-2)


def trans_ext_conv_cn():
    print('Testing first-order transient external convection...')
    # Supply file name
    file_name = os.getcwd() + '/Inputs/trans_ext_conv_cn.yaml'
    return trans_ext_conv(file_name, 3.5e-6)


def trans_ext_conv_bdf1_split():
    print('Testing first-order transient external convection...')
    # Supply file name
    file_name = os.getcwd() + '/Inputs/trans_ext_conv_bdf1_split.yaml'
    return trans_ext_conv(file_name, 1.5e-2)


def trans_ext_conv_cn_split():
    print('Testing first-order transient external convection...')
    # Supply file name
    file_name = os.getcwd() + '/Inputs/trans_ext_conv_cn_split.yaml'
    return trans_ext_conv(file_name, 8e-7)


def trans_ext_conv(file_name, e_tol):
    # Run model
    model = main_fv.lim1tr_model(file_name)
    eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, time_opts = model.run_model()

    my_t = time_opts['Run Time']
    my_mat = mat_man.get_material('A')
    C_o = bc_man.h_ext*bc_man.PA_r/(my_mat.rho*my_mat.cp)
    T_ans = bc_man.T_ext + (time_opts['T Initial'] - bc_man.T_ext)*np.exp(-1.0*C_o*my_t)
    err = np.max(np.abs(eqn_sys.T_lin - T_ans))
    if err > e_tol:
        print('\tFailed with RMSE {:0.2e}\n'.format(err))
        return 0
    else:
        print('\tPassed\n')
        return 1


def trans_end_flux_cn(plotting=False):
    print('Testing second-order transient end flux...')
    # Run model
    file_name = os.getcwd() + '/Inputs/trans_end_flux_cn.yaml'
    model = main_fv.lim1tr_model(file_name)
    eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, time_opts = model.run_model()

    # Save a few numbers
    L = np.sum(grid_man.dx_arr)*0.5
    my_t = time_opts['Run Time']
    my_mat = mat_man.get_material('A')
    alpha = my_mat.k/(my_mat.rho*my_mat.cp)
    q_in = bc_man.flux_left

    # Analytical soln (Incropera 6th edition, p. 286)
    c_one = (2*q_in/my_mat.k)*np.sqrt(alpha*my_t/np.pi)
    c_two = np.exp(-1.*grid_man.x_node**2/(4*alpha*my_t))
    c_three = q_in*grid_man.x_node/my_mat.k
    c_four = sp.special.erfc(grid_man.x_node*0.5/np.sqrt(alpha*my_t))
    T_ans = time_opts['T Initial'] + c_one*c_two - c_three*c_four

    # Calculate error
    err = np.sqrt(np.sum((T_ans - eqn_sys.T_lin)**2)/grid_man.n_tot)
    if plotting:
        is_split = ''
        if 'split' in file_name:
            is_split = '_split'
        plt.figure()
        plt.plot(grid_man.x_node, T_ans, 'o', label='Analytical')
        plt.plot(grid_man.x_node, eqn_sys.T_lin, '-', label='Numerical')
        plt.xlabel(r'Postion ($m$)')
        plt.ylabel(r'Temperature ($K$)')
        plt.legend()
        plt.title('RMSE = {:.2E}'.format(err))
        plt.savefig('./Figures/trans_end_flux_cn.png', bbox_inches='tight')
        plt.close()

    if err > 2e-4:
        print('\tFailed with RMSE {:0.2e}\n'.format(err))
        return 0
    else:
        print('\tPassed\n')
        return 1


if __name__ == '__main__':
    trans_end_conv_bdf1(plotting=True)
    trans_end_conv_cn(plotting=True)
    trans_end_conv_bdf1_split(plotting=True)
    trans_end_conv_cn_split(plotting=True)

    trans_ext_conv_bdf1()
    trans_ext_conv_cn()
    trans_ext_conv_bdf1_split()
    trans_ext_conv_cn_split()

    trans_end_flux_cn(plotting=True)
