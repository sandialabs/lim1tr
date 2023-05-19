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


class bc_base:
    def apply(self, *args):
        return 0


    def apply_operator(self, *args):
        return 0


class end_bc(bc_base):
    def __init__(self, dx_arr, my_end):
        my_end = my_end.lower()
        self.name = '{}_end'.format(my_end)
        self.dx_arr = dx_arr
        self.n_tot = dx_arr.shape[0]
        if my_end == 'left':
            self.n_ind = 0
            self.n_opp = self.n_tot - 1
            self.k_ind = 0
        elif my_end == 'right':
            self.n_ind = self.n_tot - 1
            self.n_opp = 0
            self.k_ind = self.n_tot - 2
        else:
            err_str = 'Unrecognized boundary type {}.'.format(my_end)
            raise ValueError(err_str)


class end_dirichlet(end_bc):
    def set_params(self, T):
        self.T_end = T
        self.name += '_dirichlet'


    def apply(self, eqn_sys, mat_man):
        phi = 2*mat_man.k_arr[self.k_ind]/self.dx_arr[self.n_ind]
        eqn_sys.LHS_c[self.n_ind] += phi
        eqn_sys.RHS[self.n_ind] += phi*self.T_end


class end_temperature_control(end_bc):
    def set_params(self, T_i, T_rate, T_cutoff, T_location, T_end, h_end, mint_list):
        self.T_con = T_i
        self.T_i = T_i
        self.T_rate = T_rate
        self.T_cutoff = T_cutoff
        self.name += '_control'
        self.heater_on = True
        self.h_end = h_end
        self.T_end = T_i

        # Logic for the control TC location
        if T_location == 0:
            if 'left' in self.name:
                self.cutoff_function = self.end_cutoff
            else:
                self.cutoff_function = self.opposite_cutoff
        elif T_location == len(mint_list):
            if 'right' in self.name:
                self.cutoff_function = self.end_cutoff
            else:
                self.cutoff_function = self.opposite_cutoff
        else:
            self.cutoff_function = self.interface_cutoff
            self.l_ind = mint_list[T_location - 1]
            self.r_ind = self.l_ind + 1


    def update_temperature(self, T, tot_time):
        self.T_con = self.T_i + self.T_rate*tot_time
        self.cutoff_function(T)


    def end_cutoff(self, T):
        if self.T_con >= self.T_cutoff:
            self.heater_on = False


    def opposite_cutoff(self, T):
        if self.T[self.n_opp] >= self.T_cutoff:
            self.heater_on = False


    def interface_cutoff(self, T):
        T_int = 0.5*(T[self.l_ind] + T[self.r_ind])
        if T_int >= self.T_cutoff:
            self.heater_on = False


    def apply(self, eqn_sys, mat_man):
        if self.heater_on:
            phi = 2*mat_man.k_arr[self.k_ind]/self.dx_arr[self.n_ind]
            eqn_sys.LHS_c[self.n_ind] += phi
            eqn_sys.RHS[self.n_ind] += phi*self.T_con
        else:
            phi = 2*mat_man.k_arr[self.k_ind]/self.dx_arr[self.n_ind]
            c_end = self.h_end*phi/(self.h_end + phi)
            eqn_sys.LHS_c[self.n_ind] += c_end
            eqn_sys.RHS[self.n_ind] += c_end*self.T_end


class end_convection(end_bc):
    def set_params(self, h, T):
        self.h_end = h
        self.T_end = T
        self.name += '_convection'


    def apply(self, eqn_sys, mat_man):
        '''Adds end convection BC terms to system.
        '''
        phi = 2*mat_man.k_arr[self.k_ind]/self.dx_arr[self.n_ind]
        c_end = self.h_end*phi/(self.h_end + phi)
        eqn_sys.LHS_c[self.n_ind] += c_end
        eqn_sys.RHS[self.n_ind] += c_end*self.T_end


class end_flux(end_bc):
    def set_params(self, flux):
        self.flux = flux
        self.name += '_flux'


    def apply(self, eqn_sys, mat_man):
        '''Adds heat flux bc term to end.
        '''
        eqn_sys.RHS[self.n_ind] += self.flux


class end_radiation(end_bc):
    def set_params(self, eps, T):
        self.sigma_eps = 5.67e-8*eps
        self.T_ext_4 = T**4
        self.name += '_radiation'


    def apply(self, eqn_sys, mat_man, T):
        '''Adds end radiation BC terms to system.
        '''
        eqn_sys.J_c[self.n_ind] += self.sigma_eps*4*T[self.n_ind]**3
        eqn_sys.F[self.n_ind] += self.sigma_eps*(T[self.n_ind]**4 - self.T_ext_4)


class end_radiation_arc(end_radiation):
    def set_params(self, eps, T, dTdt_max):
        super().set_params(eps, T)
        self.dTdt_max = dTdt_max
        self.T_old = 1.*T
        self.T_ext = 1.*T
        self.name += '_arc'


    def update_params(self, T, dt):
        dTdt = (T[self.n_ind] - self.T_old)/dt
        if dTdt > self.dTdt_max:
            self.T_ext = dt*self.dTdt_max + self.T_old
        else:
            self.T_ext = 1.*T[self.n_ind]
        self.T_ext_4 = self.T_ext**4


    def update_post_step(self):
        self.T_old = 1.*self.T_ext


class ext_bc(bc_base):
    def __init__(self, dx_arr, PA_r):
        self.name = 'ext'
        self.n_tot = dx_arr.shape[0]
        self.dx_PA_r = dx_arr*PA_r


class ext_convection(ext_bc):
    def set_params(self, h, T):
        self.h_ext = h
        self.T_ext = T
        self.name += '_convection'


    def apply(self, eqn_sys, mat_man):
        '''Adds external convection terms
        '''
        h_const = self.h_ext*self.dx_PA_r

        # LHS
        eqn_sys.LHS_c += h_const

        # RHS
        eqn_sys.RHS += h_const*self.T_ext


class ext_radiation(ext_bc):
    def set_params(self, eps, T):
        self.sigma_eps = 5.67e-8*eps
        self.T_ext_4 = T**4
        self.name += '_radiation'


    def apply(self, eqn_sys, mat_man, T):
        '''Adds end convection BC terms to system.
        '''
        eqn_sys.J_c += self.dx_PA_r*self.sigma_eps*4*T**3
        eqn_sys.F += self.dx_PA_r*self.sigma_eps*(T**4 - self.T_ext_4)


class timed_boundary:
    def __init__(self, bc, off_time):
        self.bc = bc
        self.off_time = off_time
        self.bc.name += '_timed'


    def set_time(self, tot_time):
        self.tot_time = tot_time


    def apply(self, *args):
        if self.off_time > self.tot_time:
            self.bc.apply(*args)
