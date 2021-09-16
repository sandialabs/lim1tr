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
import solvers
import time


class eqn_sys:
    def __init__(self, grid_man, reac_man, sol_mode, time_order, print_progress):
        self.n_tot = grid_man.n_tot
        self.dx_arr = grid_man.dx_arr
        self.print_progress = print_progress

        # Linear conduction system
        self.LHS_c = np.zeros(self.n_tot)
        self.LHS_u = np.zeros(self.n_tot)
        self.LHS_l = np.zeros(self.n_tot)
        self.RHS = np.zeros(self.n_tot)
        self.cp = np.zeros(self.n_tot)
        self.dp = np.zeros(self.n_tot)
        self.T_lin = np.zeros(self.n_tot)

        # Set solve based on solution mode
        if 'Steady' in sol_mode:
            print('SOL: Steady.')
            self.solve = self.steady_linear_solve
        elif 'Transient' in sol_mode:
            self.solve = self.transient_loop
            if 'Split' in sol_mode:
                print('SOL: Forcing split solve.')
                self.transient_solve = self.split_solve
            elif reac_man:
                if reac_man.rxn_only:
                    print('SOL: Reaction only.')
                    self.transient_solve = self.transient_ode_solve
                    if self.n_tot > 1:
                        err_str = 'Reaction Only mode not available with more than one control volume.'
                        raise ValueError(err_str)
                else:
                    print('SOL: Split solve with reactions.')
                    self.transient_solve = self.split_solve
            else:
                print('SOL: No reaction manager found. Transient linear solve.')
                self.transient_solve =  self.transient_linear_solve


    def print_sys(self):
        print(self.LHS_c)
        print(self.LHS_u)
        print(self.LHS_l)
        print(self.RHS)


    def init_linear_solver(self):
        self.my_linear_solver = solvers.tridiag


    def clean(self):
        self.LHS_c = np.zeros(self.n_tot)
        self.LHS_u = np.zeros(self.n_tot)
        self.LHS_l = np.zeros(self.n_tot)
        self.RHS = np.zeros(self.n_tot)


    def steady_linear_solve(self, mat_man, cond_man, bc_man, reac_man, data_man, t_int):
        start_time = time.time()

        # Apply conduction terms
        cond_man.apply(self, mat_man)

        # Apply boundary terms
        bc_man.apply(self, mat_man, 0)

        # Solve system
        self.my_linear_solver(self.LHS_l, self.LHS_c, self.LHS_u, 
            self.RHS, self.T_lin, self.cp, self.dp, self.n_tot)

        print('Total Solve Time: {:0.2f} s'.format(time.time() - start_time))


    def transient_loop(self, mat_man, cond_man, bc_man, reac_man, data_man, t_int):
        # Check CFL
        t_int.check_cfl(mat_man)

        # Save timing for each step
        step_time = []

        while ((t_int.end_time - t_int.tot_time) > 1e-10) and (t_int.n_step < t_int.max_steps):

            # Solve
            start_time = time.time()
            self.transient_solve(mat_man, cond_man, bc_man, reac_man, data_man, t_int)
            step_time.append(time.time() - start_time)
            if (t_int.n_step%10 == 0) & self.print_progress:
                print('{:0.1f}%\tVol Avg T: {:0.1f} K'.format(
                    100.*t_int.tot_time/t_int.end_time,
                    np.sum(t_int.T_m1*self.dx_arr)/np.sum(self.dx_arr)))

            if (t_int.n_step >= t_int.max_steps):
                print('Reached the maximum number of time steps. Exiting.')

        print('Total Solve Time: {:0.2f} s'.format(sum(step_time)))

        # Write data to a pickle
        data_man.write_data()

        # Compile data
        print('Compiling data...')
        data_man.compile_data()


    def transient_linear_solve(self, mat_man, cond_man, bc_man, reac_man, data_man, t_int):
        # Apply linear terms
        self.apply_linear_operators(mat_man, cond_man, bc_man, t_int, False)

        # Solve
        self.my_linear_solver(self.LHS_l, self.LHS_c, self.LHS_u, 
                              self.RHS, self.T_lin, self.cp, self.dp, self.n_tot)

        # Update time
        t_int.post_solve(self, self.T_lin)

        # Call data manager
        data_man.save_data(t_int, reac_man)


    def split_solve(self, mat_man, cond_man, bc_man, reac_man, data_man, t_int):
        ##############################
        ### S1 linear solve (dt/2) ###
        ##############################

        time_1_st = time.time()

        self.apply_linear_operators(mat_man, cond_man, bc_man, t_int, True)

        # Solve
        self.my_linear_solver(self.LHS_l, self.LHS_c, self.LHS_u, 
                              self.RHS, self.T_lin, self.cp, self.dp, self.n_tot)

        # Update fractional step temperature array
        t_int.T_star = np.copy(self.T_lin)

        # Reset linear system
        self.clean()

        time_1 = time.time() - time_1_st

        #########################
        ### S2 ODE solve (dt) ###
        #########################

        time_2_st = time.time()

        if reac_man:
            # Call the reaction manager and advance temperature and density
            # Return only the temperature. The reaction manager manages densities.
            t_arr = np.array([t_int.tot_time, t_int.tot_time + t_int.dt])
            t_int.T_star, err_list = reac_man.solve_ode_all_nodes(t_arr, t_int.T_star)

        time_2 = time.time() - time_2_st

        ##############################
        ### S3 linear solve (dt/2) ###
        ##############################

        time_3_st = time.time()

        self.apply_linear_operators(mat_man, cond_man, bc_man, t_int, True)

        # Solve
        self.my_linear_solver(self.LHS_l, self.LHS_c, self.LHS_u, 
            self.RHS, self.T_lin, self.cp, self.dp, self.n_tot)

        # Update temperature arrays
        t_int.post_solve(self, self.T_lin)

        # Call data manager
        data_man.save_data(t_int, reac_man)

        time_3 = time.time() - time_3_st


    def apply_linear_operators(self, mat_man, cond_man, bc_man, t_int, split_step):
        # Apply conduction terms
        cond_man.apply(self, mat_man)

        # Apply boundary terms
        bc_man.apply(self, mat_man, t_int.tot_time)

        # Apply stepper
        if(t_int.order == 1):
            t_int.apply_BDF1(self, mat_man, split_step)
        elif(t_int.order == 2):
            cond_man.apply_operator(self, mat_man, t_int.T_star)
            bc_man.apply_operator(self, mat_man, t_int.T_star, t_int.tot_time)
            t_int.apply_CN(self, mat_man, split_step)


    def transient_ode_solve(self, mat_man, cond_man, bc_man, reac_man, data_man, t_int):
        # Call the reaction manager and advance temperature and density
        # Return only the temperature. The reaction manager will manage densities.
        t_arr = np.array([t_int.tot_time, t_int.tot_time + t_int.dt])
        T_sol, err_list = reac_man.solve_ode_all_nodes(t_arr, t_int.T_star)

        # Update temperature arrays
        t_int.post_solve(self, T_sol)

        # Call data manager
        data_man.save_data(t_int, reac_man)

