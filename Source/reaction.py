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


class reaction_manager:
    def __init__(self, grid_man, other_opts):
        self.species_density = {}
        self.species_rate = {}
        self.n_tot = grid_man.n_tot
        self.mat_nodes = grid_man.mat_nodes

        # Set DSC Mode
        self.dsc_mode = 0
        if 'DSC Mode' in other_opts.keys():
            self.dsc_mode = other_opts['DSC Mode']

        # Set form of temperature ode 
        if self.dsc_mode:
            self.temperature_ode = self.linear_temperature
            if 'DSC Rate' not in other_opts.keys():
                err_str = 'Please enter a DSC Rate in the Other block'
                raise ValueError(err_str)
            self.dsc_rate = other_opts['DSC Rate']
        else:
            self.temperature_ode = self.rxn_temperature

        # Check to see if running in reaction only mode
        self.rxn_only = False
        if 'Reaction Only' in other_opts.keys():
            if other_opts['Reaction Only']:
                self.rxn_only = True

        # some constants
        self.one_third    = 1.0 / 3.0
        self.two_thirds   = 2.0 * self.one_third
        self.small_number = 1.0e-14


    def load_species(self, spec_dict, mat_man):
        # Input error checking
        if len(spec_dict['Initial Mass Fraction']) != len(spec_dict['Names']):
            err_str = 'Number of species names must match number of initial mass fractions'
            raise ValueError(err_str)

        if (abs(1. - sum(spec_dict['Initial Mass Fraction'])) > self.small_number):
            err_str = 'Initial mass fractions do not sum to 1.0'
            raise ValueError(err_str)

        # Set thermal properties
        self.mat_name = spec_dict['Material Name']
        my_mat = mat_man.get_material(self.mat_name)
        self.rho = my_mat.rho
        self.cp = my_mat.cp
        self.rho_cp = self.rho*self.cp

        # Set names, weights, and initial densities
        self.n_species = len(spec_dict['Names'])
        self.species_name_list = spec_dict['Names']
        self.molecular_weights = dict(zip(spec_dict['Names'], spec_dict['Molecular Weights']))
        
        for i in range(self.n_species):
            name = self.species_name_list[i]           
            self.species_density[name] = np.zeros(self.n_tot) + spec_dict['Initial Mass Fraction'][i]*self.rho
            self.species_rate[name] = np.zeros(self.n_tot)
        self.heat_release_rate = np.zeros(self.n_tot)
        self.temperature_rate = np.zeros(self.n_tot)


    def load_reactions(self, rxn_dict):
        self.n_rxn = len(rxn_dict.keys())
        rxn_nums = sorted(rxn_dict.keys())
        self.my_rxns = []
        self.A = np.zeros(self.n_rxn)
        self.EoR = np.zeros(self.n_rxn)
        self.H_rxn = np.zeros(self.n_rxn)

        # Build reaction system here
        #   frac_mat: converts reaction rates from total conversion to specific species
        self.frac_mat = np.zeros([self.n_species, self.n_rxn])
        self.model_list = []
        for i in range(self.n_rxn):
            rxn_info = rxn_dict[rxn_nums[i]]
            if 'Type' not in rxn_info.keys():
                rxn_info['Type'] = 'Basic'

            # Set kinetic parameters
            self.A[i] = rxn_info['A']
            self.EoR[i] = rxn_info['E']/rxn_info['R']
            self.H_rxn[i] = -1.*rxn_info['H']

            # Make reaction model
            class_ = getattr(reaction_models, reaction_models.rxn_model_dictionary[rxn_info['Type']])
            my_rxn_model = class_(rxn_info, self)

            # Build reactant map
            key_list, val_arr = my_rxn_model.build_reactant_map()
            self.frac_mat[key_list,i] -= val_arr

            # Build product map
            key_list, val_arr = my_rxn_model.build_product_map()
            self.frac_mat[key_list,i] += val_arr

            self.model_list.append(my_rxn_model)
            
        # Augment frac mat with a row for temperature (for computing the jacobian)
        self.aug_rows = np.array(self.H_rxn/self.rho_cp, ndmin=2)
        if self.dsc_mode:
            self.aug_rows = self.aug_rows*0.
        self.aug_mat = np.concatenate((self.frac_mat, self.aug_rows), axis=0)
        # print(self.aug_mat)


    def linear_temperature(self, my_r):
        '''Temperature ODE function for DSC or
        constant temperature rise'''
        return self.dsc_rate


    def rxn_temperature(self, my_r):
        '''Temperature ODE function for reaction
        source terms'''
        return np.sum(self.H_rxn*my_r)/self.rho_cp


    def evaluate_concentration_functions(self, my_v):
        '''Loop through reactions and evaluate
        the concentration function'''
        my_conc = np.zeros(self.n_rxn)
        for ii in range(self.n_rxn):
            my_conc[ii] = self.model_list[ii].concentration_function(my_v)
        return my_conc


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


    def solve_ode_all_nodes(self, t_arr, T_in, dt0=1e-8, atol=1e-6, rtol=1e-6, nsteps=5000, return_err=False):
        '''Solve the system of ODEs at each node
        This is the main function called from the transient loop
        '''
        T_out = np.zeros(self.n_tot)
        err_list = []
        for i in range(self.n_tot):
            if self.mat_name == self.mat_nodes[i]:
                # Solve system
                my_sol, nfo = self.solve_ode_node(t_arr, T_in, i, dt0=dt0, atol=atol, rtol=rtol, nsteps=nsteps)
                if return_err:
                    err_list.append(err_list)

                # Update densities
                for j in range(len(self.species_name_list)):
                    self.species_density[self.species_name_list[j]][i] = np.copy(my_sol[-1,j])

                # Get rates
                rate_arr = self.get_rates(my_sol[-1,:])

                # Update rates
                for j in range(len(self.species_name_list)):
                    self.species_rate[self.species_name_list[j]][i] = np.copy(rate_arr[j])
                self.temperature_rate[i] = np.copy(rate_arr[-1])
                self.heat_release_rate[i] = np.copy(rate_arr[-1])*self.rho_cp
                    
                # Save temperature
                T_out[i] = np.copy(my_sol[-1,-1])
            else:
                T_out[i] = np.copy(T_in[i])

        return T_out, err_list


    def solve_ode_node(self, t_arr, T_in, node_i, dt0=1e-8, atol=1e-6, rtol=1e-6, nsteps=5000):
        # Create input array
        v_in = np.zeros(self.n_species + 1)

        # Set species starting values
        for j in range(len(self.species_name_list)):
            v_in[j] = self.species_density[self.species_name_list[j]][node_i]

        # Set temperature starting value
        v_in[-1] = T_in[node_i]

        # Solve system
        sol = solve_ivp(self.evaluate_ode, (t_arr[0], t_arr[-1]),
            v_in, method='LSODA', rtol=rtol, atol=atol, jac=self.evaluate_jacobian,
            t_eval=t_arr, first_step=dt0)
        nfo = 0

        return sol.y.T, nfo


    def get_rates(self, my_sol):
        '''Given the solution of the density and temperature equations,
        return the rate at a time step.
        '''
        self.temperature_ode = self.rxn_temperature
        rate_arr = self.evaluate_ode(0., my_sol)

        if self.dsc_mode:
            self.temperature_ode = self.linear_temperature

        return rate_arr

