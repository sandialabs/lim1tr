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
    def __init__(self, grid_man, reac_man, time_opts):
        sol_mode = time_opts['Solution Mode']
        self.n_tot = grid_man.n_tot
        self.dx_arr = grid_man.dx_arr
        self.print_progress = time_opts['Print Progress']
        self.print_every = time_opts['Print Every N Steps']

        # Nonlinear options
        self.print_nonlinear = False
        self.max_nonlinear_its = 30
        self.err_tol = 1e-12

        # Linear conduction system
        self.LHS_c = np.zeros(self.n_tot)
        self.LHS_u = np.zeros(self.n_tot)
        self.LHS_l = np.zeros(self.n_tot)
        self.RHS = np.zeros(self.n_tot)
        self.cp = np.zeros(self.n_tot)
        self.dp = np.zeros(self.n_tot)
        self.T_sol = np.zeros(self.n_tot)

        # Non-linear contributions
        self.J_c = np.zeros(self.n_tot)
        self.J_u = np.zeros(self.n_tot)
        self.J_l = np.zeros(self.n_tot)
        self.F = np.zeros(self.n_tot)

        # Timers
        self.time_conduction = 0
        self.time_ode = 0
        self.time_ode_solve = 0
        self.time_ode_update = 0
        self.time_data = 0

        # Set main solver based on solution mode
        if 'Steady' in sol_mode:
            print('SOL: Steady.')
            self.solve = self.steady_solve
        elif 'Transient' in sol_mode:
            self.solve = self.transient_loop
            if 'Split' in sol_mode:
                print('SOL: Forcing split solve.')
                self.transient_solve = self.split_step_solve
            elif reac_man:
                self.init_reac(reac_man)
            else:
                print('SOL: No reaction manager found. Transient conduction solve.')
                self.transient_solve =  self.whole_step_solve


    def init_reac(self, reac_man):
        if reac_man.rxn_only:
            print('SOL: Reaction only.')
            self.transient_solve = self.transient_ode_solve
            if self.n_tot > 1:
                err_str = 'Reaction Only mode not available with more than one control volume.'
                raise ValueError(err_str)
        else:
            print('SOL: Split solve with reactions.')
            self.transient_solve = self.split_step_solve


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


    def clean_nonlinear(self):
        self.J_c = np.zeros(self.n_tot)
        self.J_u = np.zeros(self.n_tot)
        self.J_l = np.zeros(self.n_tot)
        self.F = np.zeros(self.n_tot)


    # def spitfire_solve(self):
    #     model = DiffusionReaction(ics, D, src, jac, grid_points=256)

    #     t, q = odesolve(model.right_hand_side,
    #                     model.initial_state,
    #                     stop_at_steady=True,
    #                     save_each_step=True,
    #                     linear_setup=model.setup_superlu,
    #                     linear_solve=model.solve_superlu,
    #                     step_size=PIController(target_error=1.e-8),
    #                     linear_setup_rate=20,
    #                     verbose=True,
    #                     log_rate=100,
    #                     show_solver_stats_in_situ=True)


    def transient_loop(self, mat_man, cond_man, bc_man, reac_man, data_man, t_int):
        # Check CFL
        t_int.check_cfl(mat_man)

        # Set conduction solver (linear/non-linear)
        self.set_conduction_solve(bc_man)

        # Save timing for each step
        step_time = []

        while ((t_int.end_time - t_int.tot_time) > 1e-10) and (t_int.n_step < t_int.max_steps):

            # Solve
            start_time = time.time()
            self.transient_solve(mat_man, cond_man, bc_man, reac_man, t_int)

            # Update temperature arrays
            t_int.post_solve(self.T_sol)

            # Save data
            time_data_st = time.time()
            data_man.save_data(t_int, bc_man, reac_man)
            self.time_data += time.time() - time_data_st

            step_time.append(time.time() - start_time)
            if (t_int.n_step%self.print_every == 0) & self.print_progress:
                print('{:0.1f}%\tVol Avg T: {:0.1f} K'.format(
                    100.*t_int.tot_time/t_int.end_time,
                    np.sum(t_int.T_m1*self.dx_arr)/np.sum(self.dx_arr)))

            if (t_int.n_step >= t_int.max_steps):
                print('Reached the maximum number of time steps. Exiting.')

        print('Total Solve Time: {:0.2f} s'.format(sum(step_time)))
        print('\tConduction Solve Time: {:0.2f} s'.format(self.time_conduction))
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


    def split_step_solve(self, mat_man, cond_man, bc_man, reac_man, t_int):
        ##################################
        ### S1 conduction solve (dt/2) ###
        ##################################

        self.conduction_solve(mat_man, cond_man, bc_man, t_int, True)

        # Update fractional step temperature array
        t_int.T_star = np.copy(self.T_sol)

        #########################
        ### S2 ODE solve (dt) ###
        #########################

        if reac_man:
            # Call the reaction manager and advance temperature and density
            # Return only the temperature. The reaction manager manages densities.
            time_2_st = time.time()
            t_arr = np.array([t_int.tot_time, t_int.tot_time + t_int.dt])
            t_int.T_star, err_list = reac_man.solve_ode_all_nodes(t_arr, self.T_sol)

            self.time_ode += time.time() - time_2_st
            self.time_ode_solve += reac_man.solve_ode_time
            self.time_ode_update += reac_man.update_dofs_time

        ##################################
        ### S3 conduction solve (dt/2) ###
        ##################################

        self.conduction_solve(mat_man, cond_man, bc_man, t_int, True)


    def whole_step_solve(self, mat_man, cond_man, bc_man, reac_man, t_int):
        self.conduction_solve(mat_man, cond_man, bc_man, t_int, False)


    def steady_solve(self, mat_man, cond_man, bc_man, reac_man, data_man, t_int):
        self.set_conduction_solve(bc_man)
        self.conduction_solve(mat_man, cond_man, bc_man, t_int, False)


    def set_conduction_solve(self, bc_man):
        # Set conduction solver
        if bc_man.nonlinear_flag:
            self.conduction_solve = self.nonlinear_conduction_solve
        else:
            self.conduction_solve = self.linear_conduction_solve


    def linear_conduction_solve(self, mat_man, cond_man, bc_man, t_int, split_step):
        time_st = time.time()

        # Apply linear terms
        self.apply_conduction_operators(mat_man, cond_man, bc_man, t_int, split_step)
        self.LHS_c *= mat_man.i_m_arr
        self.LHS_l *= mat_man.i_m_arr
        self.LHS_u *= mat_man.i_m_arr
        self.RHS *= mat_man.i_m_arr
        if(t_int.order == 1):
            t_int.apply_BDF1(self, mat_man, split_step)

        # Solve
        self.my_linear_solver(self.LHS_l, self.LHS_c, self.LHS_u, self.RHS,
                              self.T_sol, self.cp, self.dp, self.n_tot)

        # Reset linear system
        self.clean()

        self.time_conduction += time.time() - time_st


    def nonlinear_conduction_solve(self, mat_man, cond_man, bc_man, t_int, split_step):
        time_st = time.time()
        dT = np.zeros(self.n_tot)
        self.T_sol = np.copy(t_int.T_star)

        # Apply terms that don't depend on the new temperature
        self.apply_conduction_operators(mat_man, cond_man, bc_man, t_int, split_step)
        self.LHS_c *= mat_man.i_m_arr
        self.LHS_l *= mat_man.i_m_arr
        self.LHS_u *= mat_man.i_m_arr
        self.RHS *= mat_man.i_m_arr
        if(t_int.order == 1):
            t_int.apply_BDF1(self, mat_man, split_step)

        err = self.err_tol*2
        i = 0
        if self.print_nonlinear:
            print('Nonlinear iterations:')
        while (err > self.err_tol) & (i < self.max_nonlinear_its):
            # Build system for Newton step
            bc_man.apply_nonlinear(self, mat_man, self.T_sol)
            self.J_c *= mat_man.i_m_arr
            self.F *= mat_man.i_m_arr

            # Compute linear contributions to F
            self.F += self.LHS_c*self.T_sol - self.RHS
            self.F[:-1] += self.LHS_u[:-1]*self.T_sol[1:]
            self.F[1:] += self.LHS_l[1:]*self.T_sol[:-1]

            self.J_c += self.LHS_c
            self.J_u += self.LHS_u
            self.J_l += self.LHS_l
            self.F *= -1

            # Solve
            self.my_linear_solver(self.J_l, self.J_c, self.J_u, self.F,
                                  dT, self.cp, self.dp, self.n_tot)

            # Calculate error
            err = np.max(np.abs(dT))
            if self.print_nonlinear:
                print('\t', i, err)

            # Update
            self.T_sol += dT
            bc_man.update(self.T_sol, t_int.dt, split_step)
            i += 1

            # Reset system
            self.clean_nonlinear()

        if i >= self.max_nonlinear_its:
            print('\nWarning!!! Maximum number of nonlinear iterations reached!\n')

        bc_man.update_post_step()
        self.clean()

        self.time_conduction += time.time() - time_st


    def apply_conduction_operators(self, mat_man, cond_man, bc_man, t_int, split_step):
        # Apply conduction terms
        cond_man.apply(self, mat_man)

        # Apply boundary terms
        bc_man.apply(self, mat_man, t_int.tot_time)

        # # Apply operator for CN
        # if(t_int.order == 2):
        #     cond_man.apply_operator(self, mat_man, t_int.T_star)
        #     bc_man.apply_operator(self, mat_man, t_int.T_star, t_int.tot_time)
        #     bc_man.apply_operator_nonlinear(self, mat_man, t_int.T_star)


    # def apply_time_integration(self, mat_man, t_int, split_step):
    #     # Apply stepper
    #     if(t_int.order == 1):
    #         t_int.apply_BDF1(self, mat_man, split_step)
    #     elif(t_int.order == 2):
    #         t_int.apply_CN(self, mat_man, split_step)


    def transient_ode_solve(self, mat_man, cond_man, bc_man, reac_man, t_int):
        # Call the reaction manager and advance temperature and density
        # Return only the temperature. The reaction manager will manage densities.
        time_st = time.time()
        t_arr = np.array([t_int.tot_time, t_int.tot_time + t_int.dt])
        self.T_sol, err_list = reac_man.solve_ode_all_nodes(t_arr, t_int.T_star)

        self.time_ode += time.time() - time_st
        self.time_ode_solve += reac_man.solve_ode_time
        self.time_ode_update += reac_man.update_dofs_time
