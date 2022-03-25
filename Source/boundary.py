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
import os
import pickle as p
import boundary_types


class bc_manager:
    def __init__(self, grid_man):
        '''Applies boundary condition terms to the LHS and RHS.

        Args:
            grid_man (object): grid manager
        '''
        self.dx_arr = grid_man.dx_arr
        self.n_tot = grid_man.n_tot
        self.PA_r = grid_man.PA_r  # Perimeter to cross-sectional area ratio
        self.boundaries = []
        self.nonlinear_boundaries = []
        self.timed_boundaries = []
        self.arc_boundaries = []
        self.nonlinear_flag = False


    def setup(self, bnd_dict):
        '''Sets up parameters and apply functions for
        left, right, and external BCs
        '''

        # End BCs (left and right)
        for my_end in ['Left', 'Right']:
            end_params = bnd_dict[my_end]
            end_params['Type'] = end_params['Type'].strip().lower()
            if end_params['Type'] == 'adiabatic':
                end_bc = boundary_types.end_bc(self.dx_arr, my_end)
            elif end_params['Type'] == 'convection':
                end_bc = boundary_types.end_convection(self.dx_arr, my_end)
                end_bc.set_params(end_params['h'], end_params['T'])
            elif end_params['Type'] == 'heat flux':
                end_bc = boundary_types.end_flux(self.dx_arr, my_end)
                end_bc.set_params(end_params['Flux'])
            elif end_params['Type'] == 'radiation':
                end_bc = boundary_types.end_radiation(self.dx_arr, my_end)
                end_bc.set_params(end_params['eps'], end_params['T'])
                self.nonlinear_flag = True
            elif end_params['Type'] == 'radiation arc':
                end_bc = boundary_types.end_radiation_arc(self.dx_arr, my_end)
                end_bc.set_params(end_params['eps'], end_params['T'], end_params['Max Rate'])
                self.nonlinear_flag = True
            else:
                err_str = 'Boundary type {} for {} boundary not found.'.format(end_params['Type'], my_end)
                raise ValueError(err_str)

            self.register_bc(end_bc, end_params)

        # External BC
        ext_params = bnd_dict['External']
        ext_params['Type'] = ext_params['Type'].strip().lower()
        if ext_params['Type'] == 'adiabatic':
            ext_bc = boundary_types.ext_bc(self.dx_arr, self.PA_r)
        elif ext_params['Type'] == 'convection':
            ext_bc = boundary_types.ext_convection(self.dx_arr, self.PA_r)
            ext_bc.set_params(ext_params['h'], ext_params['T'])
        elif ext_params['Type'] == 'radiation':
            ext_bc = boundary_types.ext_radiation(self.dx_arr, self.PA_r)
            ext_bc.set_params(ext_params['eps'], ext_params['T'])
            self.nonlinear_flag = True
        else:
            err_str = 'Boundary type {} for external boundary not found.'.format(ext_params['Type'])
            raise ValueError(err_str)
        self.register_bc(ext_bc, ext_params)


    def register_bc(self, bc, params):
        if 'Deactivation Time' in params.keys():
            if 'radiation' in bc.name:
                err_str = 'Timed Radiation BC is currently not supported.'
                raise ValueError(err_str)
            off_time = params['Deactivation Time']
            timed_bc = boundary_types.timed_boundary(bc, off_time)
            self.timed_boundaries.append(timed_bc)
        elif 'radiation' in bc.name:
            self.nonlinear_boundaries.append(bc)
        else:
            self.boundaries.append(bc)

        if '_arc' in bc.name:
            self.arc_boundaries.append(bc)


    def apply(self, eqn_sys, mat_man, tot_time):
        for timed_bc in self.timed_boundaries:
            timed_bc.set_time(tot_time)
            timed_bc.apply(eqn_sys, mat_man)
        for bc in self.boundaries:
            bc.apply(eqn_sys, mat_man)


    def apply_operator(self, eqn_sys, mat_man, T, tot_time):
        for timed_bc in self.timed_boundaries:
            timed_bc.set_time(tot_time)
            timed_bc.apply_operator(eqn_sys, mat_man, T)
        for bc in self.boundaries:
            bc.apply_operator(eqn_sys, mat_man, T)


    def apply_nonlinear(self, eqn_sys, mat_man, T):
        for bc in self.nonlinear_boundaries:
            bc.apply(eqn_sys, mat_man, T)


    def apply_operator_nonlinear(self, eqn_sys, mat_man, T):
        for bc in self.nonlinear_boundaries:
            bc.apply_operator(eqn_sys, mat_man, T)


    def update(self, T, dt, split_step):
        dt_mod = 1.
        if split_step:
            dt_mod = 0.5
        for bc in self.arc_boundaries:
            bc.update_params(T, dt*dt_mod)


    def update_post_step(self):
        for bc in self.arc_boundaries:
            bc.update_post_step()


    def get_bc_output(self):
        bc_out = {}
        for bc in self.arc_boundaries:
            key = '{}_T'.format(bc.name)
            bc_out[key] = bc.T_old
        return bc_out
