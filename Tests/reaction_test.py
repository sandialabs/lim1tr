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
from scipy.special import expi
sys.path.append('../Source')
import input_parser

import matplotlib as mpl
mpl.use( 'Agg' )
import matplotlib.pyplot as plt
# latex options
plt.rc( 'text', usetex = True )
plt.rc( 'font', family = 'serif' )
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


def single_rxn_temperature_ramp(plotting=False):
    print('Testing single reaction with a constant temperature ramp...')

    # Parse file
    a_parser = input_parser.input_parser('./Inputs/single_rxn_temperature_ramp.yaml')
    mat_man, grid_man, bc_man, reac_man, data_man, time_opts = a_parser.apply_parse()

    # Solve reaction system
    n_times = 20
    t_arr = np.linspace(0, time_opts['Run Time'], n_times)
    v_in = np.zeros(reac_man.n_species + 1)
    for j in range(len(reac_man.species_name_list)):
        v_in[j] = reac_man.species_density[reac_man.species_name_list[j]][0]
    T_in = np.array([298.15])
    v_in[-1] = T_in[0]
    sol, sol_status = reac_man.reaction_systems[0].solve_ode_node(t_arr, v_in, atol=1e-13, rtol=1e-10)

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
    err = np.sum((rho_arr - sol[:,0])**2/n_times)**0.5

    if plotting:
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

    if err > 2e-7:
        print('\tFailed with RMSE {:0.2e}\n'.format(err))
        return 0
    else:
        print('\tPassed\n')
        return 1


def short_rxn():
    '''Check the short reaction evaluation at a given state for C6Li limiting
    '''
    print('Testing short circuit reaction with C6Li limiting...')
    # Parse file
    a_parser = input_parser.input_parser('./Inputs/short_only.yaml')
    mat_man, grid_man, bc_man, reac_man, data_man, time_opts = a_parser.apply_parse()

    # Build variable vector
    my_v = np.zeros(reac_man.n_species+1)
    for i in range(len(reac_man.species_name_list)):
        my_v[i] = reac_man.species_density[reac_man.species_name_list[i]][0]
    my_v[-1] = time_opts['T Initial']

    # Compute function at base inputs
    reac_sys = reac_man.reaction_systems[0]
    my_f = reac_sys.evaluate_ode(0, my_v)

    # Compute solution
    my_rxn = reac_sys.model_list[0]
    conc_C6Li = 0.13*2000./79.007
    x_C6Li = conc_C6Li*my_rxn.conc_scale
    conc_fun_1 = (np.tanh(my_rxn.t_scale*x_C6Li) + my_rxn.short_slope*x_C6Li)/(1. + my_rxn.short_slope)
    r_1 = -1.*reac_sys.A[0]*conc_fun_1
    err = np.abs(r_1*79.007/(79.007 + 90.931) - my_f[0])

    if err > 1e-12:
        print('\tFailed with RMSE {:0.2e}\n'.format(err))
        return 0
    else:
        print('\tPassed\n')
        return 1


def short_rxn_v2():
    '''Check the short reaction evaluation at a given state for CoO2 limiting
    '''
    print('Testing short circuit reaction with CoO2 limiting...')
    # Parse file
    a_parser = input_parser.input_parser('./Inputs/short_only_v2.yaml')
    mat_man, grid_man, bc_man, reac_man, data_man, time_opts = a_parser.apply_parse()

    # Build variable vector
    my_v = np.zeros(reac_man.n_species+1)
    for i in range(len(reac_man.species_name_list)):
        my_v[i] = reac_man.species_density[reac_man.species_name_list[i]][0]
    my_v[-1] = time_opts['T Initial']

    # Compute function at base inputs
    reac_sys = reac_man.reaction_systems[0]
    my_f = reac_sys.evaluate_ode(0, my_v)

    # Compute solution
    my_rxn = reac_sys.model_list[0]
    conc_CoO2 = 0.13*2000./90.931
    x_CoO2 = conc_CoO2*my_rxn.conc_scale
    conc_fun_1 = (np.tanh(my_rxn.t_scale*x_CoO2) + my_rxn.short_slope*x_CoO2)/(1. + my_rxn.short_slope)
    r_1 = -1.*reac_sys.A[0]*conc_fun_1
    err = np.abs(r_1*79.007/(79.007 + 90.931) - my_f[0])

    if err > 1e-12:
        print('\tFailed with RMSE {:0.2e}\n'.format(err))
        return 0
    else:
        print('\tPassed\n')
        return 1


def zcrit_rxn():
    '''Check the zcrit reaction evaluation and Jacobian at a given state
    '''
    print('Testing critical thickness anode reaction...')
    # Parse file
    a_parser = input_parser.input_parser('./Inputs/anode_only.yaml')
    mat_man, grid_man, bc_man, reac_man, data_man, time_opts = a_parser.apply_parse()

    # Build variable vector
    my_v = np.zeros(reac_man.n_species+1)
    for i in range(len(reac_man.species_name_list)):
        my_v[i] = reac_man.species_density[reac_man.species_name_list[i]][0]
    my_v[-1] = time_opts['T Initial']

    # Compute function at base inputs
    reac_sys = reac_man.reaction_systems[0]
    my_f = reac_sys.evaluate_ode(0, my_v)

    # Compute solution
    my_rxn = reac_sys.model_list[0]
    rho_fun = my_v[my_rxn.name_map['C6Li']]*my_v[my_rxn.name_map['EC']]/(my_rxn.rho_50 + my_v[my_rxn.name_map['EC']])
    crit_fun = np.exp(-my_rxn.C_t*my_rxn.z_c*my_v[my_rxn.name_map['Li2CO3']])
    conc_fun_1 = my_rxn.aEdges*rho_fun*crit_fun
    r_1 = -1.*reac_sys.A[0]*conc_fun_1*np.exp(-reac_sys.EoR[0]/time_opts['T Initial'])
    err = np.abs(r_1*2*79.007/(2*79.007 + 88.062) - my_f[0])

    if err > 1e-12:
        print('\tFailed with RMSE {:0.2e}\n'.format(err))
        return 0
    else:
        print('\tPassed\n')
        return 1


def fd_check(file_name, grad_check=False):
    '''Check the Jacobian computation with finite differences
    '''
    print('Checking Jacobian computation with finite differences for "{}"...'.format(file_name))
    if 'single' in file_name:
        e_thresh = 1e-7
    else:
        e_thresh = 6e-3
    np.set_printoptions(linewidth = 200)

    # Parse file
    a_parser = input_parser.input_parser('./Inputs/' + file_name + '.yaml')
    mat_man, grid_man, bc_man, reac_man, data_man, time_opts = a_parser.apply_parse()

    # Build variable vector
    my_v = np.zeros(reac_man.n_species+1)
    for i in range(len(reac_man.species_name_list)):
        my_v[i] = reac_man.species_density[reac_man.species_name_list[i]][0]
    my_v[-1] = time_opts['T Initial']

    # Get reaction system
    reac_sys = reac_man.reaction_systems[0]

    # Compute Jacobain
    jac_comp = reac_sys.evaluate_jacobian(0, my_v)

    # Compute function at base inputs
    my_f = reac_sys.evaluate_ode(0, my_v)

    # Compute Jacobain from finite differences
    if grad_check:
        print('Performing FD gradient check...')
        for up in range(1,11):
            du = 10**(-up)
            jac_fin = np.zeros(jac_comp.shape)
            for i in range(my_v.shape[0]):
                tmp_v = np.copy(my_v)
                tmp_v[i] += du
                tmp_f = reac_sys.evaluate_ode(0, tmp_v)
                jac_fin[:,i] = (tmp_f - my_f)/du
            # if up == 5:
            #     print(jac_comp)
            #     print(jac_fin)

            # Compute relative error
            rel_err = np.zeros(jac_comp.shape)
            N_v = 0
            err = 0.
            for i in range(reac_sys.n_species + 1):
                for j in range(reac_sys.n_species + 1):
                    if np.abs(jac_comp[i,j]) > 1e-15:
                        err_temp = np.abs((jac_comp[i,j] - jac_fin[i,j])/jac_comp[i,j])
                        rel_err[i,j] = err_temp
                        err += err_temp
                        N_v += 1
            # if up == 5:
            #     print(rel_err)
            print('Error: {:0.2e}, du: {:0.1e}'.format(err/N_v, du))
        return 1

    else:
        du = 10**(-5)
        jac_fin = np.zeros(jac_comp.shape)
        for i in range(my_v.shape[0]):
            tmp_v = np.copy(my_v)
            tmp_v[i] += du
            tmp_f = reac_sys.evaluate_ode(0, tmp_v)
            jac_fin[:,i] = (tmp_f - my_f)/du

        # Compute relative error
        N_v = 0
        err = 0.
        for i in range(reac_sys.n_species + 1):
            for j in range(reac_sys.n_species + 1):
                if np.abs(jac_comp[i,j]) > 1e-15:
                    err += np.abs((jac_comp[i,j] - jac_fin[i,j])/jac_comp[i,j])
                    N_v += 1
        # print('Error: {:0.2e}   {:0.1e}'.format(err/N_v, du))
        if err/N_v > e_thresh:
            print('\tFailed with Error {:0.2e}\n'.format(err/N_v))
            return 0
        else:
            print('\tPassed\n')
            return 1


if __name__ == '__main__':
    single_rxn_temperature_ramp(plotting=True)

    fd_check('jac_test')

    fd_check('jac_test_single')

    fd_check('jac_test_single', grad_check=True)

    fd_check('short_only', grad_check=True)

    fd_check('short_only_v2', grad_check=True)

    short_rxn()

    short_rxn_v2()

    fd_check('anode_only', grad_check=True)

    zcrit_rxn()
