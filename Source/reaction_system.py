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
import time


class reaction_system:
    def __init__(self, frac_mat, model_list, dsc_info):
        self.n_species, self.n_rxn = frac_mat.shape
        self.frac_mat = frac_mat
        self.model_list = model_list
        self.small_number = 1e-15

        self.dsc_mode = dsc_info[0]
        self.dsc_rate = dsc_info[1]
        if self.dsc_mode:
            self.temperature_ode = self.linear_temperature
        else:
            self.temperature_ode = self.rxn_temperature

        # Build heat of reaction array from models
        self.H_rxn = np.zeros(self.n_rxn)
        for i in range(self.n_rxn):
            self.H_rxn[i] = model_list[i].H_rxn

        # Augment frac mat with a row for temperature (for computing the jacobian)
        aug_row = np.array(self.H_rxn, ndmin=2)
        if self.dsc_mode:
            aug_row = aug_row*0.
        self.aug_mat = np.concatenate((aug_row, self.frac_mat), axis=0)


    def evaluate_ode(self, t, T_arr, species_mat):
        '''This formulation evaluates the ODEs for temperature and
        species at a set of nodes

        For DSC, dT/dt is fixed, so we use the linear temperature ODE
        that does not require division by the mass matrix
        '''
        # Calculate rate constant (n_rxn, n_nodes)
        my_k = self.evaluate_rate_constant(T_arr)

        # Calculate concentration function (n_rxn, n_nodes)
        my_conc = self.evaluate_concentration_functions(species_mat)

        # Reaction rate (n_rxn, n_nodes)
        my_r = my_k*my_conc

        # Density ODE (n_species, n_nodes)
        ds_dt = np.dot(self.frac_mat, my_r)

        # Temperature ODE (n_nodes)
        dT_dt = self.temperature_ode(my_r)

        return dT_dt, ds_dt


    def evaluate_jacobian(self, t, T_arr, species_mat):
        # Calculate rate constant (n_rxn, n_nodes)
        my_k = self.evaluate_rate_constant(T_arr)

        # Calculate concentration function for each rxn (n_rxn, n_nodes)
        my_conc = self.evaluate_concentration_functions(species_mat)

        # Derivative of reactions w.r.t all variables (n_species+1, n_rxn, n_nodes)
        dr_dv = np.zeros([self.n_species+1, self.n_rxn, T_arr.shape[0]])

        # Derivative of reactions w.r.t temperature (n_rxn, n_nodes)
        dr_dv[0,:,:] = my_conc*self.evaluate_rate_constant_derivative(T_arr, my_k)

        # Derivative of reactions w.r.t species (n_species, n_rxn, n_nodes)
        dr_dv[1:,:,:] = my_k*self.evaluate_concentration_derivatives(species_mat)

        # Compute Jacobian (n_species+1, n_species+1, n_nodes)
        d_jac = np.dot(self.aug_mat, dr_dv)

        return d_jac


    def evaluate_rate_constant(self, T_arr):
        my_k = np.zeros([self.n_rxn, T_arr.shape[0]])
        for ii in range(self.n_rxn):
            my_k[ii,:] = self.model_list[ii].evaluate_rate_constant(T_arr)
        return my_k


    def evaluate_rate_constant_derivative(self, T_arr, my_k):
        my_k_dT = np.zeros(my_k.shape)
        for ii in range(self.n_rxn):
            my_k_dT[ii,:] = self.model_list[ii].evaluate_rate_constant_derivative(T_arr, my_k[ii])
        return my_k_dT


    def evaluate_concentration_functions(self, species_mat):
        '''Loop through reactions and evaluate
        the concentration function'''
        my_conc = np.zeros([self.n_rxn, species_mat.shape[1]])
        for ii in range(self.n_rxn):
            my_conc[ii,:] = self.model_list[ii].concentration_function(species_mat)
        return my_conc


    def evaluate_concentration_derivatives(self, species_mat):
        '''Loop through reactions and evaluate
        the derivative of the concentration
        function w.r.t. each species'''
        dr_ds_part = np.zeros([species_mat.shape[0], self.n_rxn, species_mat.shape[1]])
        for ii in range(self.n_rxn):
            dr_ds_part[:,ii,:] = self.model_list[ii].concentration_derivative(species_mat)
        return dr_ds_part


    def rxn_temperature(self, my_r):
        '''Temperature ODE function for reaction
        source terms (W/m^3)'''
        return np.dot(self.H_rxn, my_r)


    def linear_temperature(self, my_r):
        '''Temperature ODE function for DSC or
        constant temperature rise (K/s)'''
        return np.full(my_r.shape[1], self.dsc_rate)






    ######### Old functions #########
    # def solve_ode_node(self, t_arr, v_in, dt0=1e-6, atol=1e-6, rtol=1e-7):
    #     # Solve system
    #     t_st = time.time()
    #     sol = solve_ivp(self.evaluate_ode, (t_arr[0], t_arr[-1]),
    #         v_in, method='LSODA', rtol=rtol, atol=atol, jac=self.evaluate_jacobian,
    #         t_eval=t_arr, first_step=dt0)
    #     if sol.status != 0:
    #         print('ODE Int Error {}'.format(sol.status))

    #     return sol.y.T, sol.status


    # def evaluate_ode(self, t, my_v):
    #     '''This formulation is an adiabatic, single volume, that advances
    #     through time.

    #     For ARC, we start at the expected initiation temperature and
    #     run an adiabatic simulation

    #     For DSC, dT/dt is fixed, so we use the linear temperature ODE
    #     '''
    #     # Calculate rate constant (array of length n_rxn)
    #     my_k = self.evaluate_rate_constant(my_v)

    #     # Calculate concentration function for each rxn
    #     my_conc = self.evaluate_concentration_functions(my_v)

    #     # Reaction rate
    #     my_r = my_k*my_conc

    #     # Density ODE array
    #     dv_dt = np.zeros(self.n_species + 1)
    #     dv_dt[:self.n_species] = np.dot(self.frac_mat, my_r)

    #     # Temperature ODE
    #     dv_dt[-1] = self.temperature_ode(my_r)

    #     return dv_dt


    # def evaluate_jacobian(self, t, my_v):
    #     # Calculate rate constant (array of length n_rxn)
    #     my_k = self.evaluate_rate_constant(my_v)

    #     # Calculate concentration function for each rxn
    #     my_conc = self.evaluate_concentration_functions(my_v)

    #     # Derivative of reactions w.r.t states
    #     dr_dv = my_k*self.evaluate_concentration_derivatives(my_v)
    #     dr_dv[self.n_species,:] = my_conc*self.evaluate_rate_constant_derivative(my_v, my_k)

    #     # Compute Jacobian
    #     d_jac = np.dot(self.aug_mat, dr_dv.T)

    #     return d_jac


    # def evaluate_rate_constant(self, my_v):
    #     my_k = np.zeros(self.n_rxn)
    #     for ii in range(self.n_rxn):
    #         my_k[ii] = self.model_list[ii].evaluate_rate_constant(my_v)
    #     return my_k


    # def evaluate_rate_constant_derivative(self, my_v, my_k):
    #     my_k_dT = np.zeros(self.n_rxn)
    #     for ii in range(self.n_rxn):
    #         my_k_dT[ii] = self.model_list[ii].evaluate_rate_constant_derivative(my_v, my_k[ii])
    #     return my_k_dT


    # def evaluate_concentration_functions(self, my_v):
    #     '''Loop through reactions and evaluate
    #     the concentration function'''
    #     my_conc = np.zeros(self.n_rxn)
    #     for ii in range(self.n_rxn):
    #         my_conc[ii] = self.model_list[ii].concentration_function(my_v)
    #     return my_conc


    # def evaluate_concentration_derivatives(self, my_v):
    #     '''Loop through reactions and evaluate
    #     the derivative of the concentration
    #     function w.r.t. each species'''
    #     dr_dv = np.zeros([self.n_species+1, self.n_rxn])
    #     for ii in range(self.n_rxn):
    #         dr_dv[:self.n_species,ii] = self.model_list[ii].concentration_derivative(my_v)
    #     return dr_dv


    # def rxn_temperature(self, my_r):
    #     '''Temperature ODE function for reaction
    #     source terms'''
    #     return np.sum(self.H_rxn*my_r)/self.rho_cp


    # def linear_temperature(self, my_r):
    #     '''Temperature ODE function for DSC or
    #     constant temperature rise'''
    #     return self.dsc_rate


    # def get_rates(self, my_sol):
    #     '''Given the solution of the density and temperature equations,
    #     return the rate at a time step.
    #     '''
    #     self.temperature_ode = self.rxn_temperature
    #     rate_arr = self.evaluate_ode(0., my_sol)

    #     if self.dsc_mode:
    #         self.temperature_ode = self.linear_temperature

    #     return rate_arr


    # def check_complete(self, my_v):
    #     '''Go through all models and check if reaction is complete
    #     or can still make progress
    #     '''
    #     is_complete = True
    #     my_conc = self.evaluate_concentration_functions(my_v)
    #     for ii in range(self.n_rxn):
    #         if my_conc[ii] > self.small_number:
    #             is_complete = False
    #             break

    #     return is_complete
