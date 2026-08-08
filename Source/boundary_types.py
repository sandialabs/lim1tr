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
    is_linear = True
    active = True       # checked by bc_manager before calling apply()
    user_name = None    # set by boundary_factory from the YAML 'Name' key

    def setup_params(self):
        return 0


    def apply(self, *args):
        raise NotImplementedError


    def post_step(self):
        return 0


    def reset_time(self, t):
        return 0


class _bc_wrapper(bc_base):
    '''Base class for BCs that wrap another BC. Forwards attribute
    lookups (my_end, n_ind, n_opp, k_ind, is_linear, etc.) to the
    wrapped BC so wrappers don't need to manually copy or special-case
    them.
    '''
    def __getattr__(self, attr):
        return getattr(self.bc, attr)


    def reset_time(self, t):
        self.bc.reset_time(t)


class end_bc(bc_base):
    yaml_type = 'adiabatic'
    required_params = []

    def __init__(self, dx_arr, my_end):
        self.my_end = my_end.lower()
        self.name = '{}_end'.format(self.my_end)
        self.dx_arr = dx_arr
        self.n_tot = dx_arr.shape[0]
        if self.my_end == 'left':
            self.n_ind = 0
            self.n_opp = self.n_tot - 1
            self.k_ind = 0
        elif self.my_end == 'right':
            self.n_ind = self.n_tot - 1
            self.n_opp = 0
            self.k_ind = self.n_tot - 2
        else:
            err_str = 'Unrecognized boundary type {}.'.format(self.my_end)
            raise ValueError(err_str)


    def apply(self, eqn_sys, mat_man, t, T_state):
        return 0


class end_dirichlet(end_bc):
    yaml_type = 'dirichlet'
    required_params = ['T']

    def setup_params(self):
        self.name += '_dirichlet'


    def apply(self, eqn_sys, mat_man, t, T_state):
        phi = 2*mat_man.k_arr[self.k_ind]/self.dx_arr[self.n_ind]
        eqn_sys.LHS_c[self.n_ind] += phi
        eqn_sys.RHS[self.n_ind] += phi*self.T


class end_convection(end_bc):
    yaml_type = 'convection'
    required_params = ['T', 'h']

    def setup_params(self):
        self.name += '_convection'


    def apply(self, eqn_sys, mat_man, t, T_state):
        '''Adds end convection BC terms to system.
        '''
        if len(self.dx_arr) == 1:
            c_end = self.h
        else:
            phi = 2*mat_man.k_arr[self.k_ind]/self.dx_arr[self.n_ind]
            c_end = self.h*phi/(self.h + phi)
        eqn_sys.LHS_c[self.n_ind] += c_end
        eqn_sys.RHS[self.n_ind] += c_end*self.T


class end_flux(end_bc):
    yaml_type = 'heat flux'
    required_params = ['flux']

    def setup_params(self):
        self.name += '_flux'


    def apply(self, eqn_sys, mat_man, t, T_state):
        '''Adds heat flux bc term to end.
        '''
        eqn_sys.RHS[self.n_ind] += self.flux


class end_radiation(end_bc):
    yaml_type = 'radiation'
    required_params = ['T', 'eps']
    is_linear = False

    def setup_params(self):
        self.sigma_eps = 5.67e-8*self.eps
        self.T_ext_4 = self.T**4
        self.name += '_radiation'


    def apply(self, eqn_sys, mat_man, t, T_state):
        '''Adds end radiation BC terms to system.
        '''
        eqn_sys.J_c[self.n_ind] += self.sigma_eps*4*T_state[self.n_ind]**3
        eqn_sys.F[self.n_ind] += self.sigma_eps*(T_state[self.n_ind]**4 - self.T_ext_4)


class ext_bc(bc_base):
    yaml_type = 'adiabatic'
    required_params = []

    def __init__(self, dx_arr, PA_r):
        self.name = 'ext'
        self.n_tot = dx_arr.shape[0]
        self.dx_PA_r = dx_arr*PA_r


    def apply(self, eqn_sys, mat_man, t, T_state):
        return 0


class ext_convection(ext_bc):
    yaml_type = 'convection'
    required_params = ['T', 'h']

    def setup_params(self):
        self.name += '_convection'


    def apply(self, eqn_sys, mat_man, t, T_state):
        '''Adds external convection terms
        '''
        h_const = self.h*self.dx_PA_r

        # LHS
        eqn_sys.LHS_c += h_const

        # RHS
        eqn_sys.RHS += h_const*self.T


class ext_radiation(ext_bc):
    yaml_type = 'radiation'
    required_params = ['T', 'eps']
    is_linear = False

    def setup_params(self):
        self.sigma_eps = 5.67e-8*self.eps
        self.T_ext_4 = self.T**4
        self.C = self.dx_PA_r*self.sigma_eps
        self.name += '_radiation'


    def apply(self, eqn_sys, mat_man, t, T_state):
        '''Adds end convection BC terms to system.
        '''
        eqn_sys.J_c += self.C*4*T_state**3
        eqn_sys.F += self.C*(T_state**4 - self.T_ext_4)


class end_temperature_control(_bc_wrapper):
    def __init__(self, bc, dx_arr, mint_list):
        self.bc = bc
        self.dx_arr = dx_arr
        self.mint_list = mint_list
        self.cutoff_trigger = False
        self.name = self.bc.name + '_control'


    def setup_params(self):
        self.conv_bc = end_convection(self.dx_arr, self.bc.my_end)
        setattr(self.conv_bc, 'T', self.T_post)
        setattr(self.conv_bc, 'h', self.h_post)
        self.conv_bc.setup_params()

        # Logic for the control TC location
        self.cutoff_function = self.end_cutoff
        if self.T_location == 0:
            if self.bc.my_end == 'left':
                self.n_cut = self.bc.n_ind
            else:
                self.n_cut = self.bc.n_opp
        elif self.T_location == len(self.mint_list):
            if self.bc.my_end == 'right':
                self.n_cut = self.bc.n_ind
            else:
                self.n_cut = self.bc.n_opp
        else:
            self.cutoff_function = self.interface_cutoff
            self.l_ind = self.mint_list[self.T_location - 1]
            self.r_ind = self.l_ind + 1


    def cutoff_base(self, T_state):
        if T_state >= self.T_cutoff:
            self.cutoff_trigger = True


    def end_cutoff(self, T_state):
        self.cutoff_base(T_state[self.n_cut])


    def interface_cutoff(self, T_state):
        self.cutoff_base(0.5*(T_state[self.l_ind] + T_state[self.r_ind]))


    def apply(self, eqn_sys, mat_man, t, T):
        self.cutoff_function(T)
        self.bc.apply(eqn_sys, mat_man, t, T)


    def post_step(self):
        if self.cutoff_trigger:
            self.apply = self.conv_bc.apply


class timed_boundary(_bc_wrapper):
    def __init__(self, bc, off_time):
        self.bc = bc
        self.off_time = off_time
        self.name = self.bc.name + '_timed'


    def apply(self, eqn_sys, mat_man, t, T_state):
        if self.off_time > t:
            self.bc.apply(eqn_sys, mat_man, t, T_state)


    def post_step(self):
        self.bc.post_step()


class temporal_boundary(_bc_wrapper):
    def __init__(self, bc):
        self.bc = bc
        self.param_names = []
        self.param_functions = []
        self.name = self.bc.name + '_temporal'
        self.time_offset = 0.0


    def add_param(self, param_name, param_function):
        self.param_names.append(param_name)
        self.param_functions.append(param_function)


    def reset_time(self, t):
        self.time_offset = t


    def apply(self, eqn_sys, mat_man, t, T_state):
        self.update_params(t - self.time_offset)
        self.bc.apply(eqn_sys, mat_man, t, T_state)


    def update_params(self, t):
        for i in range(len(self.param_names)):
            setattr(self.bc, self.param_names[i], self.param_functions[i](t))


    def post_step(self):
        self.bc.post_step()


class ramp_function:
    def __init__(self, rate, init_val):
        self.rate = rate
        self.init_val = init_val


    def __call__(self, t):
        return self.init_val + t*self.rate


class table_function:
    def __init__(self, table_x, table_y):
        self.table_x = table_x
        self.table_y = table_y


    def __call__(self, t):
        return np.interp(t, self.table_x, self.table_y)


class spacetime_table_function:
    def __init__(self, x_node, table_x, table_t, table_vals):
        from scipy.interpolate import RegularGridInterpolator
        self.x_node = x_node
        self.interp = RegularGridInterpolator((table_t, table_x), table_vals, bounds_error=False, fill_value=None)


    def __call__(self, t):
        pts = np.column_stack((np.full(self.x_node.shape, t), self.x_node))
        return self.interp(pts)
