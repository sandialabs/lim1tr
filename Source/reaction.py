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
import time
import queue
from multiprocessing import current_process
import reaction_system
import reaction_system_helper
import reaction_models
from reaction_model_factory import rxn_model_factory


class reaction_manager:
    def __init__(self, grid_man, other_opts):
        self.species_density = {}
        self.species_rate = {}
        self.n_tot = grid_man.n_tot
        self.mat_nodes = grid_man.mat_nodes
        self.first_node_list = grid_man.first_node_list
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

        # Timing arrays
        self.solve_ode_time = 0
        self.update_dofs_time = 0

        # Solver params
        self.dt0 = 1e-6
        self.atol = 1e-6
        self.rtol = 1e-7


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

        # Get number of cells and set up node key
        self.n_cells = 0
        self.cell_node_key = np.zeros(self.n_tot, dtype=int)
        self.first_node_list = self.first_node_list + [self.n_tot]
        for k in range(len(mat_man.layer_names)):
            if self.mat_name == mat_man.layer_names[k]:
                self.n_cells += 1
                self.cell_node_key[self.first_node_list[k]:self.first_node_list[k+1]] = self.n_cells

        # Set up domain where reactions are present
        self.active_nodes = np.argwhere(self.mat_nodes == self.mat_name).flatten()
        self.inactive_node_list = []


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
            my_rxn_model, frac_mat_col = rxn_model_factory(rxn_info, self.material_info)
            frac_mat[:,i] = frac_mat_col
            self.model_list.append(my_rxn_model)

            # Get active cells
            if 'Active Cells' not in rxn_info.keys():
                active_cells[i,:] = np.ones(self.n_cells, dtype=int)
            else:
                for cell_num in rxn_info['Active Cells']:
                    if cell_num > self.n_cells:
                        err_str = 'Cell {} on reaction {} does not exist. '.format(cell_num, rxn_nums[i])
                        err_str += 'Only {} cells were found in the mesh.'.format(self.n_cells)
                        raise ValueError(err_str)
                    elif cell_num < 1:
                        err_str = 'Cell number on reaction {} must be greater than 1.'.format(rxn_nums[i])
                        raise ValueError(err_str)
                    active_cells[i,cell_num-1] = 1

        # Determine the number of unique reaction systems
        self.node_to_system_map, unique_system_list = reaction_system_helper.map_all_systems(
            active_cells, self.cell_node_key)

        # Construct reaction systems by pulling out models and columns of frac_mat
        self.reaction_systems = []
        for i in range(len(unique_system_list)):
            tmp_sys = unique_system_list[i]
            rxn_inds = [j for j in range(tmp_sys.shape[0]) if tmp_sys[j]]
            model_sub_list = [self.model_list[j] for j in range(tmp_sys.shape[0]) if tmp_sys[j]]
            self.reaction_systems.append(reaction_system.reaction_system(
                frac_mat[:,rxn_inds], model_sub_list, self.rho_cp, self.dsc_info))


    def solve_ode_all_nodes(self, t_arr, T_in, dt0=1e-6, atol=1e-6, rtol=1e-7,
        processes=None, work_queue=None, result_queue=None, pool=None, n_cores=1):
        '''Solve the system of ODEs at each node
        This is the main function called from the transient loop
        '''
        T_out = np.copy(T_in)

        # Set ODE solver inputs
        self.t_arr = t_arr
        self.T_in = T_in
        self.dt0 = dt0
        self.atol = atol
        self.rtol = rtol

        # Clear nodes that were previously deactivated
        if len(self.inactive_node_list) > 0:
            self.clear_nodes()

        st_time = time.time()

        ind_list = list(range(self.active_nodes.shape[0]))
        active_node_inds = [ind_list[i::n_cores] for i in range(n_cores)]

        if n_cores == 1:
            sol_err_list = [self.solve_nodes_wrapper(active_node_inds[0])]
        elif pool is not None:
            sol_err_list = pool.map(self.solve_nodes_wrapper, active_node_inds)
        elif processes is not None:
            for act_inds in active_node_inds:
                node_data = self.node_data_wrapper(act_inds, t_arr)
                work_queue.put(node_data)
            work_queue.join()

        self.solve_ode_time = time.time() - st_time

        st_time = time.time()
        err_list = []
        if processes is not None:
            while not result_queue.empty():
                sol_err_list = result_queue.get()
                act_ind = sol_err_list[0]
                my_sol = sol_err_list[1]
                self.update_node(act_ind, my_sol)

                # Save temperature
                T_out[self.active_nodes[act_ind]] = np.copy(my_sol[-1,-1])

                err_list.append((self.active_nodes[act_ind], sol_err_list[2]))
        else:
            for core in range(n_cores):
                for k in range(len(active_node_inds[core])):
                    my_sol = sol_err_list[core][k][0]
                    act_ind = active_node_inds[core][k]
                    self.update_node(act_ind, my_sol)

                    # Save temperature
                    T_out[self.active_nodes[act_ind]] = np.copy(my_sol[-1,-1])

                    err_list.append((self.active_nodes[act_ind], sol_err_list[core][k][1]))
        self.update_dofs_time = time.time() - st_time

        return T_out, err_list


    def solve_nodes_wrapper(self, my_active_node_inds):
        sol_err_list = []

        # Loop over all active nodes
        for act_ind in my_active_node_inds:
            # Get node index
            i = self.active_nodes[act_ind]

            # Get reaction system index
            sys_ind = self.node_to_system_map[i]
            if sys_ind < 0:
                err_str = 'No reaction system specified on node {}.'.format(i)
                raise ValueError(err_str)

            # Create input array
            v_in = np.zeros(self.n_species + 1)

            # Set species starting values
            for j in range(len(self.species_name_list)):
                v_in[j] = self.species_density[self.species_name_list[j]][i]

            # Set temperature starting value
            v_in[-1] = self.T_in[i]

            # Solve system
            my_sol, my_status = self.reaction_systems[sys_ind].solve_ode_node(
                self.t_arr, v_in, dt0=self.dt0, atol=self.atol, rtol=self.rtol)

            sol_err_list.append((my_sol, my_status))

        return sol_err_list


    def node_data_wrapper(self, act_inds, t_arr):
        node_data = []
        for act_ind in act_inds:
            i = self.active_nodes[act_ind]

            # Create input array
            v_in = np.zeros(self.n_species + 1)

            # Set species starting values
            for j in range(len(self.species_name_list)):
                v_in[j] = self.species_density[self.species_name_list[j]][i]

            # Set temperature starting value
            v_in[-1] = self.T_in[i]

            # Get reaction system index
            sys_ind = self.node_to_system_map[i]
            if sys_ind < 0:
                err_str = 'No reaction system specified on node {}.'.format(i)
                raise ValueError(err_str)

            node_data.append([act_ind, v_in, t_arr, sys_ind])

        return node_data


    def solve_nodes_consumer(self, work_queue, result_queue):
        while True:
            try:
                item = work_queue.get(block=True)
            except queue.Empty:
                # print('empty', current_process().name)
                continue
            if item is None:
                print(current_process().name, 'got break')
                work_queue.task_done()
                break
            else:
                for node_data in item:
                    act_ind = node_data[0]
                    v_in = node_data[1]
                    t_arr = node_data[2]
                    sys_ind = node_data[3]

                    # Solve system
                    my_sol, my_status = self.reaction_systems[sys_ind].solve_ode_node(
                        t_arr, v_in, dt0=self.dt0, atol=self.atol, rtol=self.rtol)

                    result_queue.put([act_ind, my_sol, my_status])
                work_queue.task_done()


    def update_node(self, act_ind, my_sol):
        ''' Update densities, rates, temperature, check complete
        '''
        # Get node index
        i = self.active_nodes[act_ind]

        # Get reaction system index
        sys_ind = self.node_to_system_map[i]

        # Update densities
        for j in range(len(self.species_name_list)):
            self.species_density[self.species_name_list[j]][i] = np.copy(my_sol[-1,j])

        # Get rates
        rate_arr = self.reaction_systems[sys_ind].get_rates(my_sol[-1,:])

        # Update rates
        for j in range(len(self.species_name_list)):
            self.species_rate[self.species_name_list[j]][i] = np.copy(rate_arr[j])
        self.temperature_rate[i] = np.copy(rate_arr[-1])
        self.heat_release_rate[i] = np.copy(rate_arr[-1])*self.rho_cp

        # Check for reactant exhaustion
        is_complete = self.reaction_systems[sys_ind].check_complete(my_sol[-1,:])
        if is_complete:
            self.inactive_node_list.append(act_ind)


    def clear_nodes(self):
        '''Clear out exhausted species and rates for nodes
        who's reaction systems completed last time step.
        '''
        for act_ind in self.inactive_node_list:
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
        self.active_nodes = np.delete(self.active_nodes, self.inactive_node_list)
        self.inactive_node_list = []
