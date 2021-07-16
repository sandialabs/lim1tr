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
        self.timed_applys = active_apply_container()
        self.timed_apply_operators = active_apply_container()
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
        if 'Deactivation Time' in ext_params.keys():
            off_time = ext_params['Deactivation Time']
            self.apply_ext = active_apply_ext(self.apply_ext, off_time)
            self.timed_applys.add_function(self.apply_ext)
            self.apply_ext_operator = active_apply_ext_operator(self.apply_ext_operator, off_time)
            self.timed_apply_operators.add_function(self.apply_ext_operator)

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
        if 'Deactivation Time' in left_params.keys():
            off_time = left_params['Deactivation Time']
            self.apply_left = active_apply_end(self.apply_left, off_time)
            self.timed_applys.add_function(self.apply_left)
            self.apply_left_operator = active_apply_end_operator(self.apply_left_operator, off_time)
            self.timed_apply_operators.add_function(self.apply_left_operator)

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
        if 'Deactivation Time' in right_params.keys():
            off_time = right_params['Deactivation Time']
            self.apply_right = active_apply_end(self.apply_right, off_time)
            self.timed_applys.add_function(self.apply_right)
            self.apply_right_operator = active_apply_end_operator(self.apply_right_operator, off_time)
            self.timed_apply_operators.add_function(self.apply_right_operator)


    def apply(self, eqn_sys, mat_man, tot_time):
        self.timed_applys.set_time(tot_time)
        self.apply_ext(eqn_sys)
        self.apply_left(eqn_sys, mat_man)
        self.apply_right(eqn_sys, mat_man)


    def apply_operator(self, eqn_sys, mat_man, T, tot_time):
        self.timed_apply_operators.set_time(tot_time)
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


class active_apply_base:
    def __init__(self, my_fun, off_time):
        self.my_fun = my_fun
        self.off_time = off_time


    def set_time(self, tot_time):
        self.tot_time = tot_time


class active_apply_ext(active_apply_base):
    def __call__(self, eqn_sys):
        if self.off_time - self.tot_time > 1e-14:
            self.my_fun(eqn_sys)


class active_apply_ext_operator(active_apply_base):
    def __call__(self, eqn_sys, T):
        if self.off_time - self.tot_time > 1e-14:
            self.my_fun(eqn_sys, T)


class active_apply_end(active_apply_base):
    def __call__(self, eqn_sys, mat_man):
        if self.off_time - self.tot_time > 1e-14:
            self.my_fun(eqn_sys, mat_man)


class active_apply_end_operator(active_apply_base):
    def __call__(self, eqn_sys, mat_man, T):
        if self.off_time - self.tot_time > 1e-14:
            self.my_fun(eqn_sys, mat_man, T)


class active_apply_container:
    def __init__(self):
        self.my_functions = []


    def add_function(self, a_function):
        self.my_functions.append(a_function)


    def set_time(self, tot_time):
        for a_function in self.my_functions:
            a_function.set_time(tot_time)
