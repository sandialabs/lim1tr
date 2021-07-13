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
from scipy.integrate import solve_ivp
import reaction_models


class reaction_system:
    def __init__(self, frac_mat, model_list, rho_cp, dsc_info):
        self.n_species, self.n_rxn = frac_mat.shape
        self.frac_mat = frac_mat
        self.model_list = model_list
        self.rho_cp = rho_cp

        self.dsc_mode = dsc_info[0]
        self.dsc_rate = dsc_info[1]
        if self.dsc_mode:
            self.temperature_ode = self.linear_temperature
        else:
            self.temperature_ode = self.rxn_temperature

        # Build kinetic parameter arrays from models
        self.A = np.zeros(self.n_rxn)
        self.EoR = np.zeros(self.n_rxn)
        self.H_rxn = np.zeros(self.n_rxn)
        for i in range(self.n_rxn):
            self.A[i] = model_list[i].A
            self.EoR[i] = model_list[i].EoR
            self.H_rxn[i] = model_list[i].H_rxn

        # Augment frac mat with a row for temperature (for computing the jacobian)
        aug_row = np.array(self.H_rxn/self.rho_cp, ndmin=2)
        if self.dsc_mode:
            aug_row = aug_row*0.
        self.aug_mat = np.concatenate((self.frac_mat, aug_row), axis=0)


    def solve_ode_node(self, t_arr, v_in, dt0=1e-6, atol=1e-6, rtol=1e-6, nsteps=5000):
        # Solve system
        sol = solve_ivp(self.evaluate_ode, (t_arr[0], t_arr[-1]),
            v_in, method='LSODA', rtol=rtol, atol=atol, jac=self.evaluate_jacobian,
            t_eval=t_arr, first_step=dt0)

        return sol.y.T, sol.status


    def evaluate_ode(self, t, my_v):
        '''This formulation is an adiabatic, single volume, that advances
        through time.

        For ARC, we start at the expected initiation temperature and
        run an adiabatic simulation

        For DSC, dT/dt is fixed, so we use the linear temperature ODE
        '''
        # Calculate rate constant (array of length n_rxn)
        my_k = self.A*np.exp(-self.EoR/my_v[-1])

        # Calculate concentration function for each rxn
        my_conc = self.evaluate_concentration_functions(my_v)

        # Reaction rate
        my_r = my_k*my_conc

        # Density ODE array
        dv_dt = np.zeros(self.n_species + 1)
        dv_dt[:self.n_species] = np.dot(self.frac_mat, my_r)

        # Temperature ODE
        dv_dt[-1] = self.temperature_ode(my_r)

        return dv_dt


    def evaluate_jacobian(self, t, my_v):
        # Calculate rate constant (array of length n_rxn)
        my_k = self.A*np.exp(-self.EoR/my_v[-1])

        # Calculate concentration function for each rxn
        my_conc = self.evaluate_concentration_functions(my_v)

        # Derivative of reactions w.r.t states
        my_dr_part = np.zeros([self.n_species+1, self.n_rxn])
        for ii in range(self.n_rxn):
            my_dr_part[:self.n_species,ii] = self.model_list[ii].concentration_derivative(my_v)
        my_dr_part[self.n_species,:] = my_conc*self.EoR/my_v[-1]**2
        dr_dv = my_dr_part*my_k

        # Compute Jacobian
        d_jac = np.dot(self.aug_mat, dr_dv.T)

        return d_jac


    def evaluate_concentration_functions(self, my_v):
        '''Loop through reactions and evaluate
        the concentration function'''
        my_conc = np.zeros(self.n_rxn)
        for ii in range(self.n_rxn):
            my_conc[ii] = self.model_list[ii].concentration_function(my_v)
        return my_conc


    def rxn_temperature(self, my_r):
        '''Temperature ODE function for reaction
        source terms'''
        return np.sum(self.H_rxn*my_r)/self.rho_cp


    def linear_temperature(self, my_r):
        '''Temperature ODE function for DSC or
        constant temperature rise'''
        return self.dsc_rate


    def get_rates(self, my_sol):
        '''Given the solution of the density and temperature equations,
        return the rate at a time step.
        '''
        self.temperature_ode = self.rxn_temperature
        rate_arr = self.evaluate_ode(0., my_sol)

        if self.dsc_mode:
            self.temperature_ode = self.linear_temperature

        return rate_arr
