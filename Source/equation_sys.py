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
import multiprocessing as mp


class eqn_sys:
    def __init__(self, grid_man, reac_man, time_opts):
        sol_mode = time_opts['Solution Mode']
        self.n_cores = time_opts['Number of Cores']
        self.pool = None
        self.n_tot = grid_man.n_tot
        self.dx_arr = grid_man.dx_arr
        self.print_progress = time_opts['Print Progress']
        self.print_every = time_opts['Print Every N Steps']

        # Linear conduction system
        self.LHS_c = np.zeros(self.n_tot)
        self.LHS_u = np.zeros(self.n_tot)
        self.LHS_l = np.zeros(self.n_tot)
        self.RHS = np.zeros(self.n_tot)
        self.cp = np.zeros(self.n_tot)
        self.dp = np.zeros(self.n_tot)
        self.T_sol = np.zeros(self.n_tot)

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
                self.init_reac(reac_man.rxn_only)
            else:
                print('SOL: No reaction manager found. Transient linear solve.')
                self.transient_solve =  self.transient_linear_solve


    def init_reac(self, rxn_only):
        if rxn_only:
            print('SOL: Reaction only.')
            self.transient_solve = self.transient_ode_solve
            if self.n_tot > 1:
                err_str = 'Reaction Only mode not available with more than one control volume.'
                raise ValueError(err_str)
        else:
            print('SOL: Split solve with reactions.')
            if self.n_cores > 1:
                self.pool = mp.Pool(self.n_cores)
            self.transient_solve = self.split_solve


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
            self.RHS, self.T_sol, self.cp, self.dp, self.n_tot)

        print('Total Solve Time: {:0.2f} s'.format(time.time() - start_time))


    def transient_loop(self, mat_man, cond_man, bc_man, reac_man, data_man, t_int):
        # Check CFL
        t_int.check_cfl(mat_man)

        # Save timing for each step
        step_time = []
        self.time_diffusion = 0
        self.time_ode = 0
        self.time_ode_solve = 0
        self.time_ode_update = 0
        self.time_data = 0

        while ((t_int.end_time - t_int.tot_time) > 1e-10) and (t_int.n_step < t_int.max_steps):

            # Solve
            start_time = time.time()
            self.transient_solve(mat_man, cond_man, bc_man, reac_man, data_man, t_int)

            # Update temperature arrays
            t_int.post_solve(self, self.T_sol)

            # Save data
            time_data_st = time.time()
            data_man.save_data(t_int, reac_man)
            self.time_data += time.time() - time_data_st

            step_time.append(time.time() - start_time)
            if (t_int.n_step%self.print_every == 0) & self.print_progress:
                print('{:0.1f}%\tVol Avg T: {:0.1f} K'.format(
                    100.*t_int.tot_time/t_int.end_time,
                    np.sum(t_int.T_m1*self.dx_arr)/np.sum(self.dx_arr)))

            if (t_int.n_step >= t_int.max_steps):
                print('Reached the maximum number of time steps. Exiting.')

        if self.pool is not None:
            self.pool.close()

        print('Total Solve Time: {:0.2f} s'.format(sum(step_time)))
        print('\tDiffusion Solve Time: {:0.2f} s'.format(self.time_diffusion))
        print('\tReaction Solve Time: {:0.2f} s'.format(self.time_ode))
        print('\t\tODE Solve Time: {:0.2f} s'.format(self.time_ode_solve))
        print('\t\tODE Update Time: {:0.2f} s'.format(self.time_ode_update))
        print('\tData Storage Time: {:0.2f} s'.format(self.time_data))

        # Write data to a pickle
        time_st = time.time()
        data_man.write_data()

        # Compile data
        print('Compiling data...')
        data_man.compile_data()

        print('\tData Compiling Time: {:0.2f} s'.format(time.time() - time_st))


    def split_solve(self, mat_man, cond_man, bc_man, reac_man, data_man, t_int):
        ##################################
        ### S1 conduction solve (dt/2) ###
        ##################################

        time_1_st = time.time()

        self.apply_linear_operators(mat_man, cond_man, bc_man, t_int, True)

        # Solve
        self.my_linear_solver(self.LHS_l, self.LHS_c, self.LHS_u, 
                              self.RHS, self.T_sol, self.cp, self.dp, self.n_tot)

        # Update fractional step temperature array
        t_int.T_star = np.copy(self.T_sol)

        # Reset linear system
        self.clean()

        self.time_diffusion += time.time() - time_1_st

        #########################
        ### S2 ODE solve (dt) ###
        #########################

        if reac_man:
            # Call the reaction manager and advance temperature and density
            # Return only the temperature. The reaction manager manages densities.
            time_2_st = time.time()
            t_arr = np.array([t_int.tot_time, t_int.tot_time + t_int.dt])
            t_int.T_star, err_list = reac_man.solve_ode_all_nodes(
                t_arr, t_int.T_star, pool=self.pool, n_cores=self.n_cores)

            self.time_ode += time.time() - time_2_st
            self.time_ode_solve += reac_man.solve_ode_time
            self.time_ode_update += reac_man.update_dofs_time

        ##################################
        ### S3 conduction solve (dt/2) ###
        ##################################

        time_3_st = time.time()

        self.apply_linear_operators(mat_man, cond_man, bc_man, t_int, True)

        # Solve
        self.my_linear_solver(self.LHS_l, self.LHS_c, self.LHS_u, 
            self.RHS, self.T_sol, self.cp, self.dp, self.n_tot)

        self.time_diffusion += time.time() - time_3_st


    def transient_linear_solve(self, mat_man, cond_man, bc_man, reac_man, data_man, t_int):
        time_st = time.time()

        # Apply linear terms
        self.apply_linear_operators(mat_man, cond_man, bc_man, t_int, False)

        # Solve
        self.my_linear_solver(self.LHS_l, self.LHS_c, self.LHS_u,
                              self.RHS, self.T_sol, self.cp, self.dp, self.n_tot)

        self.time_diffusion += time.time() - time_st


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
        time_st = time.time()
        t_arr = np.array([t_int.tot_time, t_int.tot_time + t_int.dt])
        self.T_sol, err_list = reac_man.solve_ode_all_nodes(t_arr, t_int.T_star)

        self.time_ode += time.time() - time_st
        self.time_ode_solve += reac_man.solve_ode_time
        self.time_ode_update += reac_man.update_dofs_time
