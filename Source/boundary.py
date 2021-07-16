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


    def setup(self, bnd_dict):
        '''Sets up parameters and apply functions for
        left, right, and external BCs
        '''
        self.boundaries = []
        self.timed_boundaries = []

        # End BCs (left and right)
        for my_end in ['Left', 'Right']:
            my_params = bnd_dict[my_end]
            my_params['Type'] = my_params['Type'].strip().lower()
            if my_params['Type'] == 'adiabatic':
                end_bc = boundary_types.end_bc(self.dx_arr, my_end)
            elif my_params['Type'] == 'convection':
                end_bc = boundary_types.end_convection(self.dx_arr, my_end)
                end_bc.set_params(my_params['h'], my_params['T'])
            elif my_params['Type'] == 'heat flux':
                end_bc = boundary_types.end_flux(self.dx_arr, my_end)
                end_bc.set_params(my_params['Flux'])
            else:
                err_str = 'Boundary type {} for {} boundary not found.'.format(my_params['Type'], my_end)
                raise ValueError(err_str)

            if 'Deactivation Time' in my_params.keys():
                off_time = my_params['Deactivation Time']
                timed_bc = boundary_types.timed_boundary(end_bc, off_time)
                self.timed_boundaries.append(timed_bc)
            else:
                self.boundaries.append(end_bc)

        # External BC
        ext_params = bnd_dict['External']
        ext_params['Type'] = ext_params['Type'].strip().lower()
        if ext_params['Type'] == 'adiabatic':
            ext_bc = boundary_types.ext_bc(self.dx_arr, self.PA_r)
        elif ext_params['Type'] == 'convection':
            ext_bc = boundary_types.ext_convection(self.dx_arr, self.PA_r)
            ext_bc.set_params(ext_params['h'], ext_params['T'])
        else:
            err_str = 'Boundary type {} for external boundary not found.'.format(ext_params['Type'])
            raise ValueError(err_str)
        if 'Deactivation Time' in ext_params.keys():
            off_time = ext_params['Deactivation Time']
            timed_bc = boundary_types.timed_boundary(ext_bc, off_time)
            self.timed_boundaries.append(timed_bc)
        else:
            self.boundaries.append(ext_bc)


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
