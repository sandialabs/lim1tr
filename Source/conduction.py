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
        self.idx_arr = np.reciprocal( self.dx_arr ) 

        self.n_layers = grid_man.n_layers
        self.n_tot = grid_man.n_tot
        self.k_arr = np.zeros(self.n_tot - 1)

        self.internal_bounds = grid_man.internal_bounds
        self.k_bounds = grid_man.k_bounds


    def apply(self, eqn_sys, mat_man):
        '''Adds conduction terms to system.

        Args:
            eqn_sys (object): equation system object
            mat_man (object): material manager object
        '''
        # Internal nodes
        for m in range(self.n_layers):
            # Save off bounds
            bnds = self.internal_bounds[m]

            for i in range(bnds[0],bnds[1]):
                # Left (lower diagonal)
                eqn_sys.LHS_l[i] -= mat_man.k_arr[i-1]/self.dx_arr[i]

                # Center (diagonal)
                eqn_sys.LHS_c[i] += (mat_man.k_arr[i-1] + mat_man.k_arr[i])/self.dx_arr[i]

                # Right (upper diagonal)
                eqn_sys.LHS_u[i] -= mat_man.k_arr[i]/self.dx_arr[i]

        # Domain boundary nodes
        # Center (diagonal)
        eqn_sys.LHS_c[0] += mat_man.k_arr[0]/self.dx_arr[0]

        # Right (upper diagonal)
        eqn_sys.LHS_u[0] -= mat_man.k_arr[0]/self.dx_arr[0]

        # Left (lower diagonal)
        eqn_sys.LHS_l[self.n_tot-1] -= mat_man.k_arr[self.n_tot-2]/self.dx_arr[self.n_tot-1]

        # Center (diagonal)
        eqn_sys.LHS_c[self.n_tot-1] += mat_man.k_arr[self.n_tot-2]/self.dx_arr[self.n_tot-1]

        # Material interfaces
        for m in range(self.n_layers - 1):
            # Interface left node
            i_n = self.k_bounds[m][1]

            # Effective dx
            dx_e = 0.5*(self.dx_arr[i_n] + self.dx_arr[i_n+1])
            h_cont = mat_man.k_arr[i_n]/dx_e

            # Left (lower diagonal)
            eqn_sys.LHS_l[i_n] -= mat_man.k_arr[i_n-1]/self.dx_arr[i_n]

            # Center (diagonal)
            eqn_sys.LHS_c[i_n] += (mat_man.k_arr[i_n-1]/self.dx_arr[i_n]) + h_cont

            # Right (upper diagonal)
            eqn_sys.LHS_u[i_n] -= h_cont

            # Left (lower diagonal)
            eqn_sys.LHS_l[i_n+1] -= h_cont

            # Center (diagonal)
            eqn_sys.LHS_c[i_n+1] += (mat_man.k_arr[i_n+1]/self.dx_arr[i_n+1]) + h_cont

            # Right (upper diagonal)
            eqn_sys.LHS_u[i_n+1] -= mat_man.k_arr[i_n+1]/self.dx_arr[i_n+1]


    def apply_operator(self, eqn_sys, mat_man, T):
        '''Adds the action of the spatial operator on
           the temperature T to the RHS
        Args:
            eqn_sys (object) : equation system object
            mat_man (object) : material manager object
            T       (array)  : temperature at previous step
        '''        
        # Internal nodes
        for m in range(self.n_layers):
            # Save off bounds
            bnds = self.internal_bounds[m]

            # apply internal stencil
            for i in range( bnds[0], bnds[1] ):
                k_l = mat_man.k_arr[i-1]
                k_r = mat_man.k_arr[i] 
                eqn_sys.RHS[i] += ( k_l * T[i-1] - ( k_l + k_r ) * T[i] + k_r * T[i+1] ) * self.idx_arr[i]

        # Domain boundary nodes        
        n = self.n_tot-1
        eqn_sys.RHS[0] += mat_man.k_arr[0]*(T[1] - T[0])*self.idx_arr[0]
        eqn_sys.RHS[n] += mat_man.k_arr[n-1]*(T[n-1] - T[n])*self.idx_arr[n]

        # Material interfaces
        for m in range(self.n_layers - 1):
            # Interface left node
            i_n = self.k_bounds[m][1]

            # Effective dx
            dx_e = 0.5*(self.dx_arr[i_n] + self.dx_arr[i_n+1])
            h_cont = mat_man.k_arr[i_n]/dx_e

            # apply stencil
            coeff_l = mat_man.k_arr[i_n-1]/self.dx_arr[i_n]
            coeff_r = h_cont
            eqn_sys.RHS[i_n] += (coeff_l*T[i_n-1] - (coeff_l + coeff_r)*T[i_n] + coeff_r*T[i_n+1])

            coeff_l = h_cont
            coeff_r = mat_man.k_arr[i_n+1]/self.dx_arr[i_n+1]
            eqn_sys.RHS[i_n+1] += (coeff_l*T[i_n] - (coeff_l + coeff_r)*T[i_n+1] + coeff_r*T[i_n+2])
