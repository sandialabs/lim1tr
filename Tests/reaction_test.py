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
sys.path.append('../')
import main_fv
sys.path.append('../Source')
import input_parser
import reaction
import reaction_system_helper
import unit_mocks

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


def short_rxn_C6Li():
    '''Check the short reaction evaluation at a given state for C6Li limiting
    '''
    print('Testing short circuit reaction with C6Li limiting...')
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
    # rho_CoO2 = rho_CoO2_o - rate*t_arr*(90.931/(79.007 + 90.931))
    T_ans = 298.15 + dT_dt*t_arr
    t_complete = rho_C6Li_o/(rate*79.007/(79.007 + 90.931))
    T_f = 298.15 + dT_dt*t_complete
    for i in range(t_arr.shape[0]):
        if rho_C6Li[i] < 0:
            rho_C6Li[i] = 0
            T_ans[i] = T_f

    err = np.sqrt(np.mean((rho_C6Li - data_man.data_dict['C6Li'][:,0])**2))
    err += np.sqrt(np.mean((T_ans - data_man.data_dict['Temperature'][:,0])**2))

    if err > 2e-7:
        print('\tFailed with RMSE {:0.2e}\n'.format(err))
        return 0
    else:
        print('\tPassed\n')
        return 1


def short_rxn_CoO2():
    '''Check the short reaction evaluation at a given state for CoO2 limiting
    '''
    print('Testing short circuit reaction with CoO2 limiting...')
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

    if err > 1e-7:
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


def one_unique_system_test():
    print('Testing unique reaction system identification for 1 reaction system...')
    n_rxn = 3
    n_cells = 5
    active_cells = np.ones([n_rxn, n_cells], dtype=int)
    system_index, unique_system_list = reaction_system_helper.find_unique_systems(active_cells)
    true_system_index = np.zeros(n_cells, dtype=int)
    true_unique_system_list = [np.ones(n_rxn, dtype=int)]

    test_passed = verify_unique_system_test(true_system_index, true_unique_system_list,
        system_index, unique_system_list)
    if test_passed:
        print('\tPassed\n')
    return test_passed


def two_unique_system_test():
    print('Testing unique reaction system identification for 2 reaction systems...')
    n_rxn = 3
    n_cells = 5
    active_cells = np.ones([n_rxn, n_cells], dtype=int)
    active_cells[:,0] = [1,1,0]
    active_cells[:,3] = [1,1,0]
    system_index, unique_system_list = reaction_system_helper.find_unique_systems(active_cells)
    true_system_index = np.array([0,1,1,0,1], dtype=int)
    true_unique_system_list = [np.array([1,1,0], dtype=int),
        np.ones(n_rxn, dtype=int)]

    test_passed = verify_unique_system_test(true_system_index, true_unique_system_list,
        system_index, unique_system_list)
    if test_passed:
        print('\tPassed\n')
    return test_passed


def three_unique_system_test():
    print('Testing unique reaction system identification for 3 reaction systems...')
    n_rxn = 3
    n_cells = 5
    active_cells = np.ones([n_rxn, n_cells], dtype=int)
    active_cells[:,0] = [1,1,0]
    active_cells[:,3] = [1,1,0]
    active_cells[:,4] = [0,0,1]
    system_index, unique_system_list = reaction_system_helper.find_unique_systems(active_cells)
    true_system_index = np.array([0,1,1,0,2], dtype=int)
    true_unique_system_list = [np.array([1,1,0], dtype=int),
        np.ones(n_rxn, dtype=int),
        np.array([0,0,1], dtype=int)]

    test_passed = verify_unique_system_test(true_system_index, true_unique_system_list,
        system_index, unique_system_list)
    if test_passed:
        print('\tPassed\n')
    return test_passed


def verify_unique_system_test(true_index, true_list, reac_index, reac_list):
    test_passed = 1
    for i in range(true_index.shape[0]):
        if true_index[i] != reac_index[i]:
            test_passed = 0
            print('\tFailed: reaction system index incorrect.\n')
            return test_passed
    if len(true_list) != len(reac_list):
        test_passed = 0
        print('\tFailed: incorrect number of unique reactions found.\n')
        return test_passed
    for i in range(len(true_list)):
        for j in range(true_list[i].shape[0]):
            if true_list[i][j] != reac_list[i][j]:
                test_passed = 0
                print('\tFailed: incorrect unique reaction list.\n')
                return test_passed

    return test_passed


def map_system_index_to_node_test():
    print('Testing construction of the node to reaction system map...')
    n_rxn = 3
    n_cells = 4
    active_cells = np.ones([n_rxn, n_cells], dtype=int)
    active_cells[:,0] = [1,1,0]
    active_cells[:,2] = [1,1,0]
    active_cells[:,3] = [0,0,1]
    system_index, unique_system_list = reaction_system_helper.find_unique_systems(active_cells)
    cell_node_key = [0, 0, 1, 1, 2, 2, 0, 0, 3, 3, 4, 4]
    true_node_to_system_map = [-1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 2, 2]
    node_to_system_map = reaction_system_helper.map_system_to_node(system_index, cell_node_key)

    sys_diff = sum(abs(true_node_to_system_map - node_to_system_map))
    test_passed = 0
    if sys_diff > 0:
        print('\tFailed: incorrect node to system map.\n')
    else:
        print('\tPassed\n')
        test_passed = 1
    return test_passed


if __name__ == '__main__':
    single_rxn_temperature_ramp(plotting=True)

    fd_check('jac_test')

    fd_check('jac_test_single')

    fd_check('jac_test_single', grad_check=True)

    short_rxn_C6Li()

    short_rxn_CoO2()

    fd_check('anode_only', grad_check=True)

    zcrit_rxn()

    one_unique_system_test()

    two_unique_system_test()

    three_unique_system_test()

    map_system_index_to_node_test()
