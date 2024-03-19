########################################################################################
#                                                                                      #
#  Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).  #
#  Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains    #
#  certain rights in this software.                                                    #
#                                                                                      #
#  This software is released under the license detailed in the file, LICENSE.          #
#                                                                                      #
########################################################################################

import boundary_factory


class bc_manager:
    def __init__(self, grid_man):
        '''Applies boundary condition terms to the LHS and RHS.

        Args:
            grid_man (object): grid manager
        '''
        self.dx_arr = grid_man.dx_arr
        self.n_tot = grid_man.n_tot
        self.PA_r = grid_man.PA_r  # Perimeter to cross-sectional area ratio
        self.mint_list = grid_man.mint_list
        self.boundaries = []
        self.nonlinear_boundaries = []
        self.arc_boundaries = []
        self.nonlinear_flag = False


    def setup(self, bnd_dict):
        '''Sets up parameters and apply functions for
        left, right, and external BCs
        '''
        # End BCs (left and right)
        for my_end in ['Left', 'Right']:
            end_params = bnd_dict[my_end]
            end_bc = boundary_factory.factory(my_end, end_params, self.dx_arr, self.PA_r, self.mint_list)
            self.register_bc(end_bc)

        # External BC
        ext_bc = boundary_factory.factory('External', bnd_dict['External'], self.dx_arr, self.PA_r, self.mint_list)
        self.register_bc(ext_bc)


    def register_bc(self, bc):
        if 'radiation' in bc.name:
            self.nonlinear_boundaries.append(bc)
            self.nonlinear_flag = True
        else:
            self.boundaries.append(bc)


    def apply(self, eqn_sys, mat_man, t, T):
        for bc in self.boundaries:
            bc.apply(eqn_sys, mat_man, t, T)


    def apply_nonlinear(self, eqn_sys, mat_man, t, T):
        for bc in self.nonlinear_boundaries:
            bc.apply(eqn_sys, mat_man, t, T)


    def post_step(self):
        for bc in self.boundaries:
            bc.post_step()
        for bc in self.nonlinear_boundaries:
            bc.post_step()

    # Stuff below this doesn't work currently
    def update(self, T, dt):
        for bc in self.arc_boundaries:
            bc.update_params(T, dt)


    def update_post_step(self):
        for bc in self.arc_boundaries:
            bc.update_post_step()


    def get_bc_output(self):
        bc_out = {}
        for bc in self.arc_boundaries:
            key = '{}_T'.format(bc.name)
            bc_out[key] = bc.T_old
        return bc_out
