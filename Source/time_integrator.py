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


class time_int:
    def __init__(self, grid_man, time_opts):
        # order 
        self.order = time_opts['Order']

        # Saved temperature at previous two time steps
        self.T_m1 = np.zeros(grid_man.n_tot) + time_opts['T Initial']
        self.T_m2 = np.zeros(grid_man.n_tot)

        # Temperatures of previous state (split or not)
        self.T_star = np.copy(self.T_m1)

        # Time stepping
        self.dt  = time_opts['dt']
        if self.dt > 0.0:
            self.idt = 1.0 / self.dt
        self.dt_m1 = time_opts['dt']
        self.dt_list = []
        self.n_step = 1
        self.tot_time = 0.
        self.end_time = time_opts['Run Time']
        self.max_steps = time_opts['Max Steps']

        # CFL calculation parameters
        self.mat_nodes = grid_man.mat_nodes
        self.dx_arr_squared = grid_man.dx_arr**2


    def apply_BDF1(self, eqn_sys, mat_man, split_step):
        if split_step:
            self.apply_step(eqn_sys, mat_man, time_mod=2.0)
        else:
            self.apply_step(eqn_sys, mat_man)


    def apply_step(self, eqn_sys, mat_man, time_mod=1):
        # LHS
        # eqn_sys.LHS_c += mat_man.m_arr*time_mod*self.idt
        eqn_sys.LHS_c += time_mod*self.idt

        # RHS
        # eqn_sys.RHS += mat_man.m_arr*self.T_star*time_mod*self.idt
        eqn_sys.RHS += self.T_star*time_mod*self.idt


    def post_solve(self, T_sol):
        # Update time step
        self.dt_list.append(self.dt)
        self.dt_m1 = self.dt*1.
        self.tot_time += self.dt
        self.n_step += 1

        # There will be some time control here
        self.dt  = self.dt*1.
        self.idt = 1.0 / self.dt 

        # Update solution arrays
        self.T_m2 = np.copy(self.T_m1)
        self.T_m1 = np.copy(T_sol)
        self.T_star = np.copy(T_sol)


    def check_cfl(self, mat_man):
        n_tot = self.dx_arr_squared.shape[0]
        nodal_cfl = np.zeros(n_tot)
        for i in range(n_tot):
            mat_name = self.mat_nodes[i]
            nodal_cfl[i] = mat_man.get_material(mat_name).alpha*self.dt/self.dx_arr_squared[i]
        max_cfl = np.max(nodal_cfl)
        if max_cfl > 1.0:
            print('Warning: Max CFL is greater than 1')
            print('\tMax CFL: {:0.2f}'.format(max_cfl))
            print('\tSuggested maximum time step: {:0.5f}'.format(self.dt/max_cfl))
