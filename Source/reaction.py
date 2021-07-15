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

        # Get number of cells
        self.n_cells = 0
        for layer_name in mat_man.layer_names:
            if self.mat_name == layer_name:
                self.n_cells += 1

        # Set up domain where reactions are present
        self.active_nodes = np.argwhere(self.mat_nodes == self.mat_name).flatten()
        self.inactive_node_index = []


    def load_reactions(self, rxn_dict):
        self.n_rxn = len(rxn_dict.keys())
        rxn_nums = sorted(rxn_dict.keys())

        # Build reaction system here
        #   frac_mat: converts reaction rates from total conversion to specific species
        frac_mat = np.zeros([self.n_species, self.n_rxn])
        self.model_list = []
        active_cells = np.zeros([self.n_rxn, self.n_cells], dtype=int)
        for i in range(self.n_rxn):
            rxn_info = rxn_dict[rxn_nums[i]]

            # Use Basic reaction if none specified
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

            # Get active cells
            if 'Active Cells' not in rxn_info.keys():
                active_cells[i,:] = np.ones(self.n_cells, dtype=int)
            else:
                for cell_num in rxn_info['Active Cells']:
                    active_cells[i,cell_num-1] = 1

        # Determine the number of unique reaction systems
        self.find_unique_systems(active_cells)

        # Construct reaction systems by pulling out models and columns of frac_mat
        self.reaction_systems = []
        for i in range(len(self.unique_system_list)):
            tmp_sys = self.unique_system_list[i]
            rxn_inds = [j for j in range(tmp_sys.shape[0]) if tmp_sys[j]]
            model_sub_list = [self.model_list[j] for j in range(tmp_sys.shape[0]) if tmp_sys[j]]
            self.reaction_systems.append(reaction_system.reaction_system(
                frac_mat[:,rxn_inds], model_sub_list, self.rho_cp, self.dsc_info))


    def find_unique_systems(self, active_cells):
        '''Finds and indexes all unique reaction systems
        '''
        self.system_index = np.zeros(self.n_cells, dtype=int)
        self.unique_system_list = [active_cells[:,0]]
        for i in range(1, self.n_cells):
            system_exists = self.check_system_exists(active_cells[:,i])
            if system_exists:
                self.system_index[i] = self.get_system_index(active_cells[:,i])
            else:
                self.system_index[i] = len(self.unique_system_list)
                self.unique_system_list.append(active_cells[:,i])


    def check_system_exists(self, my_system):
        system_exists = False
        for a_system in self.unique_system_list:
            num_diffs = np.sum(np.abs(my_system - a_system))
            if num_diffs == 0:
                system_exists = True
                break
        return system_exists


    def get_system_index(self, my_system):
        system_index = -1
        for j in range(len(self.unique_system_list)):
            a_system = self.unique_system_list[j]
            num_diffs = np.sum(np.abs(my_system - a_system))
            if num_diffs == 0:
                system_index = j
                break
        if system_index == -1:
            err_str = 'Reaction system not found in unique system list.'
            raise ValueError(err_str)
        return system_index


    def solve_ode_all_nodes(self, t_arr, T_in, dt0=1e-6, atol=1e-6, rtol=1e-6, nsteps=5000, return_err=False):
        '''Solve the system of ODEs at each node
        This is the main function called from the transient loop
        '''
        T_out = np.copy(T_in)

        # Clear nodes that were previously deactivated
        if len(self.inactive_node_index) > 0:
            self.clear_nodes()

        # Loop over all active nodes
        err_list = []
        for act_ind in range(self.active_nodes.shape[0]):
            i = self.active_nodes[act_ind]

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

            # Check for reactant exhaustion
            is_complete = self.reaction_systems[0].check_complete(my_sol[-1,:])
            if is_complete:
                self.inactive_node_index.append(act_ind)

        return T_out, err_list


    def clear_nodes(self):
        '''Clear out exhausted species and rates for nodes
        who's reaction systems completed last time step.
        '''
        for act_ind in self.inactive_node_index:
            i = self.active_nodes[act_ind]
            # Clear exhausted species
            for j in range(len(self.species_name_list)):
                if self.species_density[self.species_name_list[j]][i] < self.small_number:
                    self.species_density[self.species_name_list[j]][i] = 0.0

            # Clear rates
            for j in range(len(self.species_name_list)):
                self.species_rate[self.species_name_list[j]][i] = 0.0
            self.temperature_rate[i] = 0.0
            self.heat_release_rate[i] = 0.0

        # Remove nodes from active nodes
        self.active_nodes = np.delete(self.active_nodes, self.inactive_node_index)
        self.inactive_node_index = []
