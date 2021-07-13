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
import reaction_system
import reaction_models


class reaction_manager:
    def __init__(self, grid_man, other_opts):
        self.species_density = {}
        self.species_rate = {}
        self.n_tot = grid_man.n_tot
        self.mat_nodes = grid_man.mat_nodes
        self.cross_area = other_opts['Y Dimension']*other_opts['Z Dimension']

        # Set DSC Mode
        self.dsc_mode = 0
        self.dsc_rate = 0
        if 'DSC Mode' in other_opts.keys():
            self.dsc_mode = other_opts['DSC Mode']

        # Set form of temperature ode
        if self.dsc_mode:
            if 'DSC Rate' not in other_opts.keys():
                err_str = 'Please enter a DSC Rate in the Other block'
                raise ValueError(err_str)
            self.dsc_rate = other_opts['DSC Rate']
        self.dsc_info = (self.dsc_mode, self.dsc_rate)

        # Check to see if running in reaction only mode
        self.rxn_only = False
        if 'Reaction Only' in other_opts.keys():
            if other_opts['Reaction Only']:
                self.rxn_only = True
                if grid_man.n_tot != 1:
                    err_str = 'Multiple control volumes found in reaction only simulation.\n'
                    err_str += 'Check that dx is equal to the domain length.'
                    raise ValueError(err_str)

        # Small constant
        self.small_number = 1.0e-15


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
        molecular_weights = dict(zip(spec_dict['Names'], spec_dict['Molecular Weights']))

        for i in range(self.n_species):
            name = self.species_name_list[i]
            self.species_density[name] = np.zeros(self.n_tot)
            self.species_rate[name] = np.zeros(self.n_tot)
            for j in range(self.n_tot):
                if self.mat_name == self.mat_nodes[j]:
                    self.species_density[name][j] = spec_dict['Initial Mass Fraction'][i]*self.rho
        self.heat_release_rate = np.zeros(self.n_tot)
        self.temperature_rate = np.zeros(self.n_tot)

        # Dictionary for parsed material info
        self.material_info = {}
        self.material_info['Names'] = self.species_name_list
        self.material_info['Molecular Weights'] = molecular_weights
        self.material_info['rho'] = self.rho
        self.material_info['cp'] = self.cp


    def load_reactions(self, rxn_dict):
        self.n_rxn = len(rxn_dict.keys())
        rxn_nums = sorted(rxn_dict.keys())

        # Build reaction system here
        #   frac_mat: converts reaction rates from total conversion to specific species
        frac_mat = np.zeros([self.n_species, self.n_rxn])
        self.model_list = []
        for i in range(self.n_rxn):
            rxn_info = rxn_dict[rxn_nums[i]]
            if 'Type' not in rxn_info.keys():
                rxn_info['Type'] = 'Basic'

            # Make reaction model
            class_ = getattr(reaction_models, reaction_models.rxn_model_dictionary[rxn_info['Type']])
            my_rxn_model = class_(rxn_info, self.material_info)

            # Build reactant map
            key_list, val_arr = my_rxn_model.build_reactant_map()
            frac_mat[key_list,i] -= val_arr

            # Build product map
            key_list, val_arr = my_rxn_model.build_product_map()
            frac_mat[key_list,i] += val_arr

            self.model_list.append(my_rxn_model)

        # Determine the number of unique reaction systems

        # Construct reaction systems by pulling out models and columns of frac_mat
        self.reaction_systems = []
        self.reaction_systems.append(reaction_system.reaction_system(
            frac_mat, self.model_list, self.rho_cp, self.dsc_info))


    def solve_ode_all_nodes(self, t_arr, T_in, dt0=1e-6, atol=1e-6, rtol=1e-6, nsteps=5000, return_err=False):
        '''Solve the system of ODEs at each node
        This is the main function called from the transient loop
        '''
        T_out = np.zeros(self.n_tot)
        err_list = []
        for i in range(self.n_tot):
            if self.mat_name == self.mat_nodes[i]:
                # Create input array
                v_in = np.zeros(self.n_species + 1)

                # Set species starting values
                for j in range(len(self.species_name_list)):
                    v_in[j] = self.species_density[self.species_name_list[j]][i]

                # Set temperature starting value
                v_in[-1] = T_in[i]

                # Solve system
                my_sol, my_status = self.reaction_systems[0].solve_ode_node(
                    t_arr, v_in, dt0=dt0, atol=atol, rtol=rtol, nsteps=nsteps)
                if return_err:
                    err_list.append(my_status)

                # Update densities
                for j in range(len(self.species_name_list)):
                    self.species_density[self.species_name_list[j]][i] = np.copy(my_sol[-1,j])

                # Get rates
                rate_arr = self.reaction_systems[0].get_rates(my_sol[-1,:])

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
