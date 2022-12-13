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
from scipy.sparse import csc_matrix, diags, bmat, eye as speye
from scipy.sparse.linalg import splu as superlu_factor


class eqn_sys:
    def __init__(self, mat_man, cond_man, bc_man, grid_man, reac_man, time_opts):
        sol_mode = time_opts['Solution Mode']
        self.n_tot = grid_man.n_tot
        self.dof_node = 1
        self.dx_arr = grid_man.dx_arr

        # Time options
        self.initial_state = time_opts['T Initial']
        self.norm_weighting = np.full(self.n_tot, 1e-3)
        self.end_time = time_opts['Run Time']
        self.fixed_step = time_opts['Fixed Step']
        self.dt = time_opts['dt']
        self.print_progress = time_opts['Print Progress']
        self.print_every = time_opts['Print Every N Steps']

        # Diffusion and material operators
        self.mat_man = mat_man
        self.cond_man = cond_man
        self.bc_man = bc_man

        # Reaction manager
        self.reac_man = reac_man
        if self.reac_man:
            self.dof_node += self.reac_man.n_species

            # Build initial state
            self.initial_state = np.hstack([
                self.initial_state, self.reac_man.initial_density])
            species_weight = 1/self.reac_man.material_info['rho']
            self.norm_weighting = np.hstack([
                self.norm_weighting, 
                np.full(self.reac_man.initial_density.shape[0], species_weight)])

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

        # Timers and counters
        self.time_conduction = 0
        self.time_reaction = 0
        self.time_conduction_jac = 0
        self.time_reaction_jac = 0
        self.factor_superlu_time = 0
        self.solve_superlu_time = 0
        self.cond_apply_time = 0
        self.cond_F_time = 0
        self.bc_time = 0
        self.nlbc_time = 0
        self.clean_time = 0
        self.time_ode = 0
        self.time_ode_solve = 0
        self.time_ode_update = 0
        self.time_data = 0
        self.rhs_count = 0
        self.setup_count = 0
        self.solve_count = 0

        # Create operator and identity
        self._lhs_inverse_operator = None
        self._I = csc_matrix(speye(self.n_tot*self.dof_node))

        # Initialize Jacobian
        ones = np.ones(self.n_tot)
        diag_ind = np.arange(self.n_tot, dtype=int)
        T_jac = csc_matrix(diags([ones[1:], ones, ones[:-1]], [-1, 0, 1]))

        if self.reac_man:
            T_jac_inds = csc_matrix(diags([
                diag_ind[1:],
                diag_ind[:-1] + self.n_tot],
                [-1, 1], dtype=int))
            R_jac_inds = np.zeros([self.dof_node, self.dof_node, self.n_tot], dtype=int)
            R_flat_inds = np.arange(self.dof_node*self.dof_node*self.n_tot, dtype=int) + 2*self.n_tot
            R_jac_all_inds = R_flat_inds.reshape([self.dof_node, self.dof_node, self.n_tot])

            # Assemble full Jacobian
            R_jac = np.zeros([self.dof_node, self.dof_node, self.n_tot])
            for i in range(self.reac_man.n_cells):
                b1, b2 = self.reac_man.cells[i].bounds
                R_jac[:,:,b1:b2] = 1.0
                R_jac_inds[:,:,b1:b2] = R_jac_all_inds[:,:,b1:b2]
            R_jac_inds[0,0,:] = R_jac_all_inds[0,0,:]
            blocks = []
            ind_blocks = []
            for i in range(self.dof_node):
                row = []
                ind_row = []
                for j in range(self.dof_node):
                    row.append(csc_matrix(diags(R_jac[i,j,:])))
                    ind_row.append(csc_matrix(diags(R_jac_inds[i,j,:], dtype=int)))
                blocks.append(row)
                ind_blocks.append(ind_row)
            blocks[0][0] += T_jac
            ind_blocks[0][0] += T_jac_inds
            self.jac = bmat(blocks, format='csc')
            self.ind_jac = bmat(ind_blocks, format='csc')
        else:
            self.jac = T_jac
            self.ind_jac = csc_matrix(diags([
                diag_ind[1:],
                diag_ind + 2*self.n_tot,
                diag_ind[:-1] + self.n_tot],
                [-1, 0, 1], dtype=int))


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


    def right_hand_side(self, t, state):
        self.rhs_count += 1
        t_st = time.time()
        self.clean()
        self.clean_nonlinear()
        self.clean_time += time.time() - t_st

        t_st = time.time()
        # Assemble conduction RHS
        tc_st = time.time()
        self.cond_man.apply(self, self.mat_man)
        self.cond_apply_time += time.time() - tc_st

        # Apply linear boundary terms
        tbc_st = time.time()
        self.bc_man.apply(self, self.mat_man, t)
        self.bc_time += time.time() - tbc_st

        # Compute linear contributions to F
        tl_st = time.time()
        T_arr = state[:self.n_tot]
        self.F = self.LHS_c*T_arr - self.RHS
        self.F[:-1] += self.LHS_u[:-1]*T_arr[1:]
        self.F[1:] += self.LHS_l[1:]*T_arr[:-1]
        self.cond_F_time += time.time() - tl_st

        # Non-linear BCs
        tnl_st = time.time()
        self.bc_man.apply_nonlinear(self, self.mat_man, T_arr)
        self.nlbc_time += time.time() - tnl_st

        self.F *= -1*self.mat_man.i_m_arr

        self.time_conduction += time.time() - t_st

        # Assemble RXN F
        if self.reac_man:
            t_st = time.time()
            RHS_T, RHS_species = self.reac_man.right_hand_side(t, state)
            self.F += RHS_T*self.mat_man.i_rcp
            self.F = np.hstack([self.F, RHS_species])
            self.time_reaction += time.time() - t_st

        return self.F


    def setup_superlu(self, t, state, prefactor):
        self.setup_count += 1
        t_st = time.time()
        self.clean()
        self.clean_nonlinear()
        self.clean_time += time.time() - t_st

        # Assemble conduction RHS
        t_st = time.time()
        self.cond_man.apply(self, self.mat_man)

        # Apply linear boundary terms
        self.bc_man.apply(self, self.mat_man, t)

        # Non-linear BCs
        self.bc_man.apply_nonlinear(self, self.mat_man, state[:self.n_tot])

        self.J_c += self.LHS_c
        self.J_u += self.LHS_u
        self.J_l += self.LHS_l
        self.J_c *= -1*self.mat_man.i_m_arr
        self.J_l *= -1*self.mat_man.i_m_arr
        self.J_u *= -1*self.mat_man.i_m_arr

        # Make flat array to map in to Jacobian
        J_flat = np.zeros(self.n_tot*(2 + self.dof_node**2))
        J_flat[:self.n_tot] = self.J_l
        J_flat[self.n_tot:2*self.n_tot] = self.J_u
        self.time_conduction_jac += time.time() - t_st

        t_st = time.time()
        if self.reac_man:
            # Assemble RXN Jacobian
            R_jac = self.reac_man.jacobian(t, state)

            # Convert temperature ODEs to K/s
            for j in range(self.dof_node):
                R_jac[0,j,:] *= self.mat_man.i_rcp

            # Add in conduction contribution to center diagonal
            R_jac[0,0,:] += self.J_c

            # Flatten Jacobian
            J_flat[2*self.n_tot:] = R_jac.ravel()
        else:
            J_flat[2*self.n_tot:] = self.J_c
        self.jac.data = J_flat[self.ind_jac.data]
        self.time_reaction_jac += time.time() - t_st

        t_st = time.time()
        self._lhs_inverse_operator = superlu_factor(prefactor * self.jac - self._I)
        self.factor_superlu_time += time.time() - t_st


    def solve_superlu(self, residual):
        self.solve_count += 1
        t_st = time.time()
        l_op_solve = self._lhs_inverse_operator.solve(residual)
        self.solve_superlu_time += time.time() - t_st
        return l_op_solve, 1, True







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
