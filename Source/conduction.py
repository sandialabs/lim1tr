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


class conduction_manager:
    def __init__(self, grid_man):
        '''Applies conduction terms to the LHS. 
        The grid stuff in initialize will probably
        change to be handled by a grid manager.
        Sets up bounds and loads grid.

        Args:
            grid_man (object): grid manager
        '''
        self.dx_arr  = grid_man.dx_arr
        self.idx_e = np.reciprocal(0.5*(self.dx_arr[:-1] + self.dx_arr[1:]))
        self.n_tot = grid_man.n_tot


    def apply(self, eqn_sys, mat_man):
        '''Adds conduction terms to system.

        Args:
            eqn_sys (object): equation system object
            mat_man (object): material manager object
        '''
        # Loop over faces
        for i in range(self.n_tot-1):
            h_face = mat_man.k_arr[i]*self.idx_e[i]

            # Left node
            eqn_sys.LHS_c[i] += h_face
            eqn_sys.LHS_u[i] -= h_face

            # Right node
            eqn_sys.LHS_l[i+1] -= h_face
            eqn_sys.LHS_c[i+1] += h_face


    def apply_operator(self, eqn_sys, mat_man, T):
        '''Adds the action of the spatial operator on
           the temperature T to the RHS
        Args:
            eqn_sys (object) : equation system object
            mat_man (object) : material manager object
            T       (array)  : temperature at previous step
        '''
        # Loop over faces
        for i in range(self.n_tot-1):
            h_face = mat_man.k_arr[i]*self.idx_e[i]

            # Left node
            eqn_sys.RHS[i] += h_face*(T[i+1] - T[i])

            # Right node
            eqn_sys.RHS[i+1] += h_face*(T[i] - T[i+1])
