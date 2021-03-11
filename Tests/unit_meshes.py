########################################################################################
#                                                                                      #
#  Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).  #
#  Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains    #
#  certain rights in this software.                                                    #
#                                                                                      #
#  This software is released under the license detailed in the file, LICENSE.          #
#                                                                                      #
########################################################################################

import numpy as np
import sys
sys.path.append('../Source')
import material
import grid


def mesh_one(dx_a=1.,rho_a=1,cp_a=1.,k_a=1.):
    '''Ten node, one material mesh.
    '''
    mat_man = material.material_manager()
    dx_arr = np.zeros(10) + dx_a
    mat_nodes = np.asarray(['A']*10)
    mint_list = [9]
    oth_dict = {}
    oth_dict['Y Dimension'] = 1.
    oth_dict['Z Dimension'] = 1.
    grid_man = grid.grid_manager()
    grid_man.setup_grid(dx_arr, mat_nodes, mint_list, oth_dict)
    mat_man.add_mesh(grid_man)
    a_mat = material.fv_material('A')
    a_mat.set_rho(rho_a)
    a_mat.set_cp(cp_a)
    a_mat.set_k(k_a)
    mat_man.add_material(a_mat, 'A')
    mat_man.eval_props()

    return grid_man, mat_man


def mesh_two(dx_a=1.,rho_a=1,cp_a=1.,k_a=1.,dx_b=1.,rho_b=1,cp_b=1.,k_b=1.):
    '''Ten node, two material mesh.
    '''
    mat_man = material.material_manager()
    dx_arr = np.zeros(10)
    dx_arr[:5] = dx_a
    dx_arr[5:] = dx_b
    mat_nodes = np.asarray(['A']*5 + ['B']*5)
    mint_list = [4, 9]    
    oth_dict = {}
    oth_dict['Y Dimension'] = 1.
    oth_dict['Z Dimension'] = 1.
    grid_man = grid.grid_manager()
    grid_man.setup_grid(dx_arr, mat_nodes, mint_list, oth_dict)
    mat_man.cont_res = np.zeros(grid_man.n_mats-1)
    mat_man.add_mesh(grid_man)
    a_mat = material.fv_material('A')
    a_mat.set_rho(rho_a)
    a_mat.set_cp(cp_a)
    a_mat.set_k(k_a)
    b_mat = material.fv_material('B')
    b_mat.set_rho(rho_b)
    b_mat.set_cp(cp_b)
    b_mat.set_k(k_b)
    mat_man.add_material(a_mat, 'A')
    mat_man.add_material(b_mat, 'B')
    mat_man.eval_props()

    return grid_man, mat_man
