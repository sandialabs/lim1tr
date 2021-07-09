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

        self.n_mats = grid_man.n_mats
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
        for m in range(self.n_mats):
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
        for m in range(self.n_mats - 1):
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
        for m in range(self.n_mats):
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
        for m in range(self.n_mats - 1):
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


class bc_manager:
    def __init__(self, grid_man):
        '''Applies boundary condition terms to the LHS and RHS.

        Args:
            grid_man (object): grid manager
        '''
        self.dx_arr = grid_man.dx_arr
        self.n_tot = grid_man.n_tot
        self.PA_r = 1.  # Perimeter to cross-sectional area ratio


    def setup(self, bnd_dict):
        '''Sets up parameters and apply functions for
        left, right, and external BCs
        '''
        ext_params = bnd_dict['External']
        ext_params['Type'] = ext_params['Type'].strip().lower()
        if ext_params['Type'] == 'adiabatic':
            self.apply_ext = self.apply_adiabatic
            self.apply_ext_operator = self.apply_adiabatic_operator
        elif ext_params['Type'] == 'convection':
            self.apply_ext = self.apply_external_convection
            self.apply_ext_operator = self.apply_external_convection_operator
            self.h_ext = ext_params['h']
            self.T_ext = ext_params['T']
        else:
            err_str = 'Boundary type {} for external boundary not found.'.format(ext_params['Type'])
            raise ValueError(err_str)

        left_params = bnd_dict['Left']
        left_params['Type'] = left_params['Type'].strip().lower()
        if left_params['Type'] == 'adiabatic':
            self.apply_left = self.apply_adiabatic
            self.apply_left_operator = self.apply_adiabatic_operator
        elif left_params['Type'] == 'convection':
            self.apply_left = self.apply_left_convection
            self.apply_left_operator = self.apply_left_convection_operator
            self.h_left = left_params['h']
            self.T_left = left_params['T']
        elif left_params['Type'] == 'heat flux':
            self.apply_left = self.apply_left_flux
            self.apply_left_operator = self.apply_left_flux_operator
            self.flux_left = left_params['Flux']
        else:
            err_str = 'Boundary type {} for left boundary not found.'.format(left_params['Type'])
            raise ValueError(err_str)

        right_params = bnd_dict['Right']
        right_params['Type'] = right_params['Type'].strip().lower()
        if right_params['Type'] == 'adiabatic':
            self.apply_right = self.apply_adiabatic
            self.apply_right_operator = self.apply_adiabatic_operator
        elif right_params['Type'] == 'convection':
            self.apply_right = self.apply_right_convection
            self.apply_right_operator = self.apply_right_convection_operator
            self.h_right = right_params['h']
            self.T_right = right_params['T']
        elif right_params['Type'] == 'heat flux':
            self.apply_right = self.apply_right_flux
            self.apply_right_operator = self.apply_right_flux_operator
            self.flux_right = right_params['Flux']
        else:
            err_str = 'Boundary type {} for right boundary not found.'.format(right_params['Type'])
            raise ValueError(err_str)


    def apply(self, eqn_sys, mat_man):
        self.apply_ext(eqn_sys)
        self.apply_left(eqn_sys, mat_man)
        self.apply_right(eqn_sys, mat_man)


    def apply_operator(self, eqn_sys, mat_man, T):
        self.apply_ext_operator(eqn_sys, T)
        self.apply_left_operator(eqn_sys, mat_man, T)
        self.apply_right_operator(eqn_sys, mat_man, T)


    def apply_adiabatic(self, *args):
        return 0


    def apply_adiabatic_operator(self, *args):
        return 0


    def apply_left_convection(self, eqn_sys, mat_man):
        '''Adds left end convection BC terms to system.
        '''
        phi_left = 2*mat_man.k_arr[0]/self.dx_arr[0]
        c_left = self.h_left*phi_left/(self.h_left + phi_left)
        eqn_sys.LHS_c[0] += c_left
        eqn_sys.RHS[0] += c_left*self.T_left


    def apply_left_convection_operator(self, eqn_sys, mat_man, T):
        '''Adds the action of the left end convection
        terms on the previous time step to the RHS
        '''
        phi_left = 2*mat_man.k_arr[0]/self.dx_arr[0]
        c_left = self.h_left*phi_left/(self.h_left + phi_left)
        eqn_sys.RHS[0] += c_left*(self.T_left - T[0])


    def apply_right_convection(self, eqn_sys, mat_man):
        '''Adds right end convection BC terms to system.
        '''
        phi_right = 2*mat_man.k_arr[self.n_tot-2]/self.dx_arr[self.n_tot-1]
        c_right = self.h_right*phi_right/(self.h_right + phi_right)
        eqn_sys.LHS_c[self.n_tot-1] += c_right
        eqn_sys.RHS[self.n_tot-1] += c_right*self.T_right


    def apply_right_convection_operator(self, eqn_sys, mat_man, T):
        '''Adds the action of the right end convection
        terms on the previous time step to the RHS
        '''
        phi_right = 2*mat_man.k_arr[self.n_tot-2]/self.dx_arr[self.n_tot-1]
        c_right = self.h_right*phi_right/(self.h_right + phi_right)
        eqn_sys.RHS[self.n_tot-1] += c_right*(self.T_right - T[self.n_tot-1])


    def apply_left_flux(self, eqn_sys, *args):
        '''Adds heat flux bc term to left end.
        '''
        eqn_sys.RHS[0] += self.flux_left


    def apply_left_flux_operator(self, eqn_sys, *args):
        '''Adds action of left heat flux bc terms
        on the previous time step to the RHS
        '''
        eqn_sys.RHS[0] += self.flux_left


    def apply_right_flux(self, eqn_sys, *args):
        '''Adds heat flux bc term to right end.
        '''
        eqn_sys.RHS[self.n_tot-1] += self.flux_right


    def apply_right_flux_operator(self, eqn_sys, *args):
        '''Adds action of right heat flux bc terms
        on the previous time step to the RHS
        '''
        eqn_sys.RHS[self.n_tot-1] += self.flux_right


    def apply_external_convection(self, eqn_sys):
        '''Adds external convection terms
        '''
        for i in range(self.n_tot):
            h_const = self.h_ext*self.dx_arr[i]*self.PA_r

            # LHS
            eqn_sys.LHS_c[i] += h_const

            # RHS
            eqn_sys.RHS[i] += h_const*self.T_ext


    def apply_external_convection_operator(self, eqn_sys, T):
        '''Adds the action of the external convection terms
        on the previous time step to the RHS
        '''
        for i in range(self.n_tot):
            # convection constant
            h_const = self.h_ext*self.dx_arr[i]*self.PA_r

            # RHS
            eqn_sys.RHS[i] += h_const*(self.T_ext - T[i])

