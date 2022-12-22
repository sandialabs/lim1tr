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

        self.con_time = np.zeros(self.n_rxn)
        self.con_d_time = np.zeros(self.n_rxn)
        self.rate_time = np.zeros(self.n_rxn)
        self.rate_d_time = np.zeros(self.n_rxn)


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
            t_st = time.time()
            my_k[ii,:] = self.model_list[ii].evaluate_rate_constant(T_arr)
            self.rate_time[ii] += time.time() - t_st
        return my_k


    def evaluate_rate_constant_derivative(self, T_arr, my_k):
        my_k_dT = np.zeros(my_k.shape)
        for ii in range(self.n_rxn):
            t_st = time.time()
            my_k_dT[ii,:] = self.model_list[ii].evaluate_rate_constant_derivative(T_arr, my_k[ii])
            self.rate_d_time[ii] += time.time() - t_st
        return my_k_dT


    def evaluate_concentration_functions(self, species_mat):
        '''Loop through reactions and evaluate
        the concentration function'''
        my_conc = np.zeros([self.n_rxn, species_mat.shape[1]])
        for ii in range(self.n_rxn):
            t_st = time.time()
            my_conc[ii,:] = self.model_list[ii].concentration_function(species_mat)
            self.con_time[ii] += time.time() - t_st
        return my_conc


    def evaluate_concentration_derivatives(self, species_mat):
        '''Loop through reactions and evaluate
        the derivative of the concentration
        function w.r.t. each species'''
        dr_ds_part = np.zeros([species_mat.shape[0], self.n_rxn, species_mat.shape[1]])
        for ii in range(self.n_rxn):
            t_st = time.time()
            dr_ds_part[:,ii,:] = self.model_list[ii].concentration_derivative(species_mat)
            self.con_d_time[ii] += time.time() - t_st
        return dr_ds_part


    def rxn_temperature(self, my_r):
        '''Temperature ODE function for reaction
        source terms (W/m^3)'''
        return np.dot(self.H_rxn, my_r)


    def linear_temperature(self, my_r):
        '''Temperature ODE function for DSC or
        constant temperature rise (K/s)'''
        return np.full(my_r.shape[1], self.dsc_rate)
