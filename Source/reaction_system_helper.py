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


def find_unique_systems(active_cells):
    '''Finds and indexes all unique reaction systems
    '''
    n_cells = active_cells.shape[1]
    system_index = np.zeros(n_cells, dtype=int)
    unique_system_list = [active_cells[:,0]]
    for i in range(1, n_cells):
        system_exists = check_system_exists(active_cells[:,i], unique_system_list)
        if system_exists:
            system_index[i] = get_system_index(active_cells[:,i], unique_system_list)
        else:
            system_index[i] = len(unique_system_list)
            unique_system_list.append(active_cells[:,i])
    return system_index, unique_system_list


def check_system_exists(my_system, unique_system_list):
    system_exists = False
    for a_system in unique_system_list:
        num_diffs = np.sum(np.abs(my_system - a_system))
        if num_diffs == 0:
            system_exists = True
            break
    return system_exists


def get_system_index(my_system, unique_system_list):
    system_index = -1
    for j in range(len(unique_system_list)):
        a_system = unique_system_list[j]
        num_diffs = np.sum(np.abs(my_system - a_system))
        if num_diffs == 0:
            system_index = j
            break
    if system_index == -1:
        err_str = 'Reaction system not found in unique system list.'
        raise ValueError(err_str)
    return system_index
