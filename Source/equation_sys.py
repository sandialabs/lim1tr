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
        self.target_error = time_opts['Target Error']
        self.linear_setup_rate = time_opts['Maximum Steps Per Jacobian']
        self.end_time = time_opts['Run Time']
        self.fixed_step = time_opts['Fixed Step']
        self.dt = time_opts['dt']
        if self.fixed_step:
            self.t = np.arange(0, self.end_time, self.dt)
        else:
            dt_f = 1.0/time_opts['Output Frequency']
            self.t = np.arange(0, self.end_time, dt_f)
        self.t = np.concatenate((self.t, np.full(1,self.end_time)))

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

        # Steady non-linear options
        self.print_nonlinear = False
        self.max_nonlinear_its = 30
        self.err_tol = 1e-8

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

        self.J_fac_c = np.zeros(self.n_tot)
        self.J_fac_l = np.zeros(self.n_tot)
        self.J_fac_u = np.zeros(self.n_tot)
        self.dT = np.zeros(self.n_tot)

        # Timers and counters
        self.time_conduction = 0
        self.time_reaction = 0
        self.time_conduction_jac = 0
        self.time_reaction_jac = 0
        self.factor_superlu_time = 0
        self.solve_linear_time = 0
        self.cond_apply_time = 0
        self.cond_F_time = 0
        self.bc_time = 0
        self.nlbc_time = 0
        self.clean_time = 0

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


    def init_tridiag_solver(self):
        self.tridiag_solver = solvers.tridiag


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

        self.conduction_right_hand_side(t, state)

        # Assemble RXN F
        if self.reac_man:
            t_st = time.time()
            RHS_T, RHS_species = self.reac_man.right_hand_side(t, state)
            if self.reac_man.dsc_mode:
                self.F += RHS_T
            else:
                self.F += RHS_T*self.mat_man.i_rcp
            self.F = np.hstack([self.F, RHS_species])
            self.time_reaction += time.time() - t_st

        return self.F


    def conduction_right_hand_side(self, t, state):
        t_st = time.time()

        # Linear energy contributions
        T_state = state[:self.n_tot]
        self.apply_energy_eq_operators(t, T_state)

        # Compute linear contributions to F
        tl_st = time.time()
        self.F += self.LHS_c*T_state - self.RHS
        self.F[:-1] += self.LHS_u[:-1]*T_state[1:]
        self.F[1:] += self.LHS_l[1:]*T_state[:-1]
        self.cond_F_time += time.time() - tl_st

        # Non-linear BCs
        tnl_st = time.time()
        self.bc_man.apply_nonlinear(self, self.mat_man, t, T_state)
        self.nlbc_time += time.time() - tnl_st

        # Transient contribution
        self.F *= -1*self.mat_man.i_m_arr

        self.time_conduction += time.time() - t_st


    def setup_jacobian(self, t, state):
        # Assemble conduction terms
        self.setup_conduction_jacobian(t, state)

        # Make flat array to map in to Jacobian for adding species equations
        t_st = time.time()
        J_flat = np.zeros(self.n_tot*(2 + self.dof_node**2))
        J_flat[:self.n_tot] = self.J_l
        J_flat[self.n_tot:2*self.n_tot] = self.J_u

        if self.reac_man:
            # Assemble RXN Jacobian
            R_jac = self.reac_man.jacobian(t, state)

            # Convert temperature ODEs to K/s
            if not self.reac_man.dsc_mode:
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


    def setup_conduction_jacobian(self, t, state):
        t_st = time.time()

        # # Apply linear energy terms
        T_state = state[:self.n_tot]
        self.apply_energy_eq_operators(t, T_state)

        # Non-linear BCs
        self.bc_man.apply_nonlinear(self, self.mat_man, t, state[:self.n_tot])

        self.J_c += self.LHS_c
        self.J_u += self.LHS_u
        self.J_l += self.LHS_l
        self.J_c *= -1*self.mat_man.i_m_arr
        self.J_l *= -1*self.mat_man.i_m_arr
        self.J_u *= -1*self.mat_man.i_m_arr
        self.time_conduction_jac += time.time() - t_st


    def apply_energy_eq_operators(self, t, T_state):
        # Apply conduction terms
        tc_st = time.time()
        self.cond_man.apply(self, self.mat_man)
        self.cond_apply_time += time.time() - tc_st

        # Apply linear boundary terms
        tbc_st = time.time()
        self.bc_man.apply(self, self.mat_man, t, T_state)
        self.bc_time += time.time() - tbc_st


    def setup_superlu(self, t, state, prefactor):
        self.setup_count += 1
        t_st = time.time()
        self.clean()
        self.clean_nonlinear()
        self.clean_time += time.time() - t_st

        self.setup_jacobian(t, state)

        t_st = time.time()
        self._lhs_inverse_operator = superlu_factor(prefactor * self.jac - self._I)
        self.factor_superlu_time += time.time() - t_st


    def solve_superlu(self, residual):
        self.solve_count += 1
        t_st = time.time()
        l_op_solve = self._lhs_inverse_operator.solve(residual)
        self.solve_linear_time += time.time() - t_st
        return l_op_solve, 1, True


    def setup_conduction(self, t, state, prefactor):
        self.setup_count += 1
        t_st = time.time()
        self.clean()
        self.clean_nonlinear()
        self.clean_time += time.time() - t_st

        self.setup_conduction_jacobian(t, state)

        self.J_fac_c = prefactor*self.J_c - 1
        self.J_fac_l = prefactor*self.J_l
        self.J_fac_u = prefactor*self.J_u


    def solve_conduction(self, residual):
        self.solve_count += 1
        t_st = time.time()
        self.tridiag_solver(self.J_fac_l, self.J_fac_c, self.J_fac_u, residual,
                            self.dT, self.cp, self.dp, self.n_tot)
        self.solve_linear_time += time.time() - t_st
        return self.dT, 1, True


    def post_step(self, t, state, residual, number_of_time_steps):
        self.bc_man.post_step()


    def steady_solve(self):
        self.set_conduction_solve()
        self.conduction_solve(0, self.initial_state)


    def set_conduction_solve(self):
        # Set conduction solver
        if self.bc_man.nonlinear_flag:
            self.conduction_solve = self.nonlinear_conduction_solve
        else:
            self.conduction_solve = self.linear_conduction_solve


    def linear_conduction_solve(self, t, state):
        time_st = time.time()

        # Apply linear energy terms
        self.apply_energy_eq_operators(t, state)
        self.LHS_c *= self.mat_man.i_m_arr
        self.LHS_l *= self.mat_man.i_m_arr
        self.LHS_u *= self.mat_man.i_m_arr
        self.RHS *= self.mat_man.i_m_arr
        self.F = self.LHS_c*state - self.RHS
        self.F[:-1] += self.LHS_u[:-1]*state[1:]
        self.F[1:] += self.LHS_l[1:]*state[:-1]

        # Solve
        self.tridiag_solver(self.LHS_l, self.LHS_c, self.LHS_u, self.F,
                              self.T_sol, self.cp, self.dp, self.n_tot)
        self.T_sol = state - self.T_sol

        # Reset linear system
        self.clean()

        self.time_conduction += time.time() - time_st


    def nonlinear_conduction_solve(self, t, state):
        time_st = time.time()
        dT = np.zeros(self.n_tot)
        self.T_sol = np.copy(state)

        # Apply terms that don't depend on the new temperature
        T_state = state[:self.n_tot]
        self.apply_energy_eq_operators(t, T_state)
        self.LHS_c *= self.mat_man.i_m_arr
        self.LHS_l *= self.mat_man.i_m_arr
        self.LHS_u *= self.mat_man.i_m_arr
        self.RHS *= self.mat_man.i_m_arr

        err = self.err_tol*2
        i = 0
        if self.print_nonlinear:
            print('Nonlinear iterations:')
        while (err > self.err_tol) & (i < self.max_nonlinear_its):
            # Build system for Newton step
            self.bc_man.apply_nonlinear(self, self.mat_man, t, self.T_sol)
            self.J_c *= self.mat_man.i_m_arr
            self.F *= self.mat_man.i_m_arr

            # Compute linear contributions to F
            self.F += self.LHS_c*self.T_sol - self.RHS
            self.F[:-1] += self.LHS_u[:-1]*self.T_sol[1:]
            self.F[1:] += self.LHS_l[1:]*self.T_sol[:-1]

            self.J_c += self.LHS_c
            self.J_u += self.LHS_u
            self.J_l += self.LHS_l
            self.F *= -1

            # Solve
            self.tridiag_solver(self.J_l, self.J_c, self.J_u, self.F,
                                  dT, self.cp, self.dp, self.n_tot)

            # Calculate error
            err = np.max(np.abs(dT))
            if self.print_nonlinear:
                print('\t', i, err)

            # Update
            self.T_sol += dT
            i += 1

            # Reset system
            self.clean_nonlinear()

        if i >= self.max_nonlinear_its:
            print('\nWarning!!! Maximum number of nonlinear iterations reached!\n')

        self.clean()

        self.time_conduction += time.time() - time_st


    def print_statistics(self):
        print('LIM1TR Stastics:')
        print('  RHS Assembly')
        print(f'- Conduction (s)    : {self.time_conduction:0.3f}')
        print(f'  - Apply (s)       : {self.cond_apply_time:0.3f}')
        print(f'  - BC Apply (s)    : {self.bc_time:0.3f}')
        print(f'  - Linear Parts (s): {self.cond_F_time:0.3f}')
        print(f'  - NL Parts (s)    : {self.nlbc_time:0.3f}')
        if self.reac_man:
            print(f'- Reaction (s)      : {self.time_reaction:0.3f}')
        print(f'- Calls             : {self.rhs_count}')

        print('\n  Jacobian Assembly')
        print(f'- Conduction (s)    : {self.time_conduction_jac:0.3f}')
        if self.reac_man:
            print(f'- Reaction (s)      : {self.time_reaction_jac:0.3f}')
        print(f'- Factor SuperLU (s): {self.factor_superlu_time:0.3f}')
        print(f'- Calls             : {self.setup_count}')

        print(f'\n  Solve Linear System')
        print(f'- Time (s): {self.solve_linear_time:0.3f}')
        print(f'- Calls   : {self.solve_count}')

        print('\n  Other')
        if self.reac_man:
            slice_time = 0.0
            for ii in range(self.reac_man.n_cells):
                slice_time += self.reac_man.cells[ii].slice_time
            print(f'- Slice (s): {slice_time:0.3f}')
        print(f'- Clean (s): {self.clean_time:0.3f}\n')
