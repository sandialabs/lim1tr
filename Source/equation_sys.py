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
from scipy.sparse import csc_matrix, diags, block_diag, eye as speye
from scipy.sparse.linalg import splu as superlu_factor

from spitfire import PIController, odesolve
from spitfire import RK4ClassicalS4P4, BackwardEulerS1P1Q1, SimpleNewtonSolver 


class eqn_sys:
    def __init__(self, grid_man, reac_man, time_opts):
        sol_mode = time_opts['Solution Mode']
        self.n_tot = grid_man.n_tot
        self.dx_arr = grid_man.dx_arr

        # Time options
        self.T_init = time_opts['T Initial']
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

        # Create operator and identity
        self._lhs_inverse_operator = None
        self._I = csc_matrix(speye(self.n_tot))

        # Set main solver based on solution mode
        if 'Steady' in sol_mode:
            print('SOL: Steady.')
            self.solve = self.steady_solve
        elif 'Transient' in sol_mode:
            # self.solve = self.transient_loop
            self.solve = self.spitfire_solve
            # if 'Split' in sol_mode:
            #     print('SOL: Forcing split solve.')
            #     self.transient_solve = self.split_step_solve
            # elif reac_man:
            #     self.init_reac(reac_man)
            # else:
            #     print('SOL: No reaction manager found. Transient conduction solve.')
            #     self.transient_solve =  self.whole_step_solve


    # def init_reac(self, reac_man):
    #     if reac_man.rxn_only:
    #         print('SOL: Reaction only.')
    #         self.transient_solve = self.transient_ode_solve
    #         if self.n_tot > 1:
    #             err_str = 'Reaction Only mode not available with more than one control volume.'
    #             raise ValueError(err_str)
    #     else:
    #         print('SOL: Split solve with reactions.')
    #         self.transient_solve = self.split_step_solve


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


    def spitfire_solve(self, mat_man, cond_man, bc_man, reac_man, data_man, t_int):
        self.mat_man = mat_man
        self.cond_man = cond_man
        self.bc_man = bc_man

        t, q = odesolve(self.right_hand_side,
                    t_int.T_star,
                    stop_at_time=t_int.end_time,
                    save_each_step=True,
                    linear_setup=self.setup_superlu,
                    linear_solve=self.solve_superlu,
                    step_size=PIController(target_error=1.e-8),
                    linear_setup_rate=20,
                    verbose=True,
                    log_rate=100,
                    show_solver_stats_in_situ=True)
        # t, q = odesolve(self.right_hand_side,
        #                 t_int.T_star,
        #                 stop_at_time=t_int.end_time,
        #                 save_each_step=True,
        #                 step_size=t_int.dt,
        #                 method=RK4ClassicalS4P4())
        # t, q = odesolve(self.right_hand_side,
        #                 self.T_init,
        #                 stop_at_time=t_int.end_time,
        #                 save_each_step=True,
        #                 linear_setup=self.setup_superlu,
        #                 linear_solve=self.solve_superlu,
        #                 step_size=t_int.dt,
        #                 verbose=True,
        #                 method=BackwardEulerS1P1Q1(SimpleNewtonSolver()))
        self.T_sol = q[-1,:]


    def right_hand_side(self, t, state):
        self.clean()
        self.clean_nonlinear()
        # print(state)
        # Assemble conduction RHS
        self.cond_man.apply(self, self.mat_man)

        # Apply linear boundary terms
        self.bc_man.apply(self, self.mat_man, t)

        # Compute linear contributions to F
        self.F = self.LHS_c*state - self.RHS
        self.F[:-1] += self.LHS_u[:-1]*state[1:]
        self.F[1:] += self.LHS_l[1:]*state[:-1]

        # Non-linear BCs
        self.bc_man.apply_nonlinear(self, self.mat_man, state)
        self.F *= -1*self.mat_man.i_m_arr

        # Assemble RXN F

        return self.F


    def setup_superlu(self, t, state, prefactor):
        self.clean()
        self.clean_nonlinear()

        # Assemble conduction RHS
        self.cond_man.apply(self, self.mat_man)

        # Apply linear boundary terms
        self.bc_man.apply(self, self.mat_man, t)

        # Non-linear BCs
        self.bc_man.apply_nonlinear(self, self.mat_man, state)

        self.J_c += self.LHS_c
        self.J_u += self.LHS_u
        self.J_l += self.LHS_l
        self.J_c *= -1*self.mat_man.i_m_arr
        self.J_l *= -1*self.mat_man.i_m_arr
        self.J_u *= -1*self.mat_man.i_m_arr

        # Assemble RXN Jacobian


        # Assemble full Jacobian
        jac = csc_matrix(diags([self.J_l[1:], self.J_c, self.J_u[:-1]], [-1, 0, 1]))
        # print(jac.toarray())
        self._lhs_inverse_operator = superlu_factor(prefactor * jac - self._I)


    def solve_superlu(self, residual):
        return self._lhs_inverse_operator.solve(residual), 1, True




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


    def transient_ode_solve(self, mat_man, cond_man, bc_man, reac_man, t_int):
        # Call the reaction manager and advance temperature and density
        # Return only the temperature. The reaction manager will manage densities.
        time_st = time.time()
        t_arr = np.array([t_int.tot_time, t_int.tot_time + t_int.dt])
        self.T_sol, err_list = reac_man.solve_ode_all_nodes(t_arr, t_int.T_star)

        self.time_ode += time.time() - time_st
        self.time_ode_solve += reac_man.solve_ode_time
        self.time_ode_update += reac_man.update_dofs_time
