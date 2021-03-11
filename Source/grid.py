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


class grid_manager:
    def setup_grid(self, dx_arr, mat_nodes, mint_list, oth_dict):
        '''Sets up the grid.

        Args:
            dx_arr (numpy array): width of each cv in x direction
            mat_nodes (numpy array): material name at each node
            mint_list (list): list of left node at each interface (and last node)
            oth_dict (dictionary): other options (for y and z dimensions)
        '''
        self.dx_arr = dx_arr
        self.dv_arr = dx_arr*oth_dict['Y Dimension']*oth_dict['Z Dimension']
        self.mat_nodes = mat_nodes
        self.mint_list = mint_list
        self.n_mats = len(mint_list)
        self.n_tot = mat_nodes.shape[0]

        # Build bounds lists
        self.internal_bounds = [[1, mint_list[0]]]
        self.k_bounds = [[0, mint_list[0]]]
        for m in range(1,self.n_mats):
            self.internal_bounds.append([mint_list[m-1]+2, mint_list[m]])
            self.k_bounds.append([mint_list[m-1]+1, mint_list[m]])

        # Build node locations
        self.x_node = np.zeros(self.n_tot)
        for i in range(self.n_tot):
            self.x_node[i] = 0.5*self.dx_arr[i] + np.sum(self.dx_arr[:i])
