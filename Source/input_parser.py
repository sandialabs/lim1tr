########################################################################################
#                                                                                      #
#  Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).  #
#  Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains    #
#  certain rights in this software.                                                    #
#                                                                                      #
#  This software is released under the license detailed in the file, LICENSE.          #
#                                                                                      #
########################################################################################

import warnings
import numpy as np
import os
import logging
import yaml
import material
import boundary
import grid
import reaction
import data
import events


class input_parser:
    def __init__(self, i_file):
        '''Input file parser. This is called first.

        Args:
            i_file (str): yaml input file name
        '''
        self.i_file = i_file
        with open(self.i_file, 'r') as f:
            self.cap_dict = yaml.load(f, Loader=yaml.FullLoader)

        # Get the folder and file names for saving output
        abs_path = os.path.abspath(self.i_file)
        self.fold_name, file_name = os.path.split(abs_path)
        self.file_name = file_name.split('.yaml')[0]


    def print_dictionary(self):
        for key in self.cap_dict.keys():
            logging.info(key)
            logging.info(self.cap_dict[key])


    def apply_parse(self):
        '''Parses a file

        Returns:
            mat_man (object): material manager object
            grid_man (object): grid manager
            bc_man (object): boundary condition manager
        '''
        # List of required blocks
        cap_list = ['Materials', 'Domain Table', 'Boundary', 'Time', 'Other']

        # Check that all required blocks are present
        missing_blocks = []
        for block in cap_list:
            if block not in self.cap_dict.keys():
                missing_blocks.append(block)
        if len(missing_blocks) > 0:
            err_str = 'The following blocks were not found in the input file:\n\t{}'.format(missing_blocks)
            raise ValueError(err_str)

        # Domain table
        grid_man = grid.grid_manager()
        self.load_table(grid_man)
        grid_man.setup_grid()
        grid_man.set_PA_r(self.cap_dict['Other'])

        # Materials
        mat_man = material.material_manager()
        self.load_materials(mat_man, grid_man)

        # Time
        time_opts = self.load_time(grid_man)

        # Boundaries
        bc_man = boundary.bc_manager(grid_man)
        bc_man.setup(self.cap_dict['Boundary'])

        # Parse optional reaction blocks
        # All the parsing will be handled by reaction manager so that it
        # can be a stand-alone system that solves ODEs on a single CV.
        if ('Reactions' not in self.cap_dict.keys()) != ('Species' not in self.cap_dict.keys()):
            if 'Reactions' not in self.cap_dict.keys():
                err_str = 'The Reaction block must accompany the Species block'
            else:
                err_str = 'The Species block must accompany the Reaction block'
            raise ValueError(err_str)
        elif ('Species' in self.cap_dict.keys()) and ('Reactions' in self.cap_dict.keys()):
            # Set up reaction manager
            reac_man = reaction.reaction_manager(grid_man, self.cap_dict['Other'])

            # Initialize species
            reac_man.load_species(self.cap_dict['Species'], mat_man)

            # Initialize reaction parameter
            reac_man.load_reactions(self.cap_dict['Reactions'])

        else:
            reac_man = False

        # Validate DSC mode input
        if reac_man and reac_man.dsc_mode:
            n_mats = len(self.cap_dict['Materials'])
            if n_mats > 1:
                warnings.warn(
                    'DSC mode: {} materials specified; only 1 is expected.'.format(n_mats))
            n_layers = len(self.cap_dict['Domain Table']['Material Name'])
            if n_layers > 1:
                raise ValueError(
                    'DSC mode: domain table has {} rows; only 1 is expected.'.format(n_layers))
            bnd = self.cap_dict['Boundary']
            for loc in ('Left', 'Right'):
                bc_type = bnd.get(loc, {}).get('Type', 'Adiabatic')
                if bc_type.lower() != 'adiabatic':
                    raise ValueError(
                        'DSC mode: {} boundary is {} (expected Adiabatic).'.format(loc, bc_type))
            ext = bnd.get('External')
            if ext is not None:
                ext_list = ext if isinstance(ext, list) else [ext]
                for bc in ext_list:
                    bc_type = bc.get('Type', 'Adiabatic')
                    if bc_type.lower() != 'adiabatic':
                        raise ValueError(
                            'DSC mode: External boundary is {} (expected Adiabatic).'.format(bc_type))

        # Events (optional)
        self.load_events(bc_man, grid_man, reac_man)

        # Data manager
        data_man = data.data_manager(grid_man, reac_man, self.cap_dict, self.fold_name, self.file_name)

        return mat_man, grid_man, bc_man, reac_man, data_man, time_opts


    def load_events(self, bc_man, grid_man, reac_man):
        '''Parse the optional Events block into an event manager on bc_man.

        Events is a mapping of rule name to rule. Each rule pairs a condition
        (checked every accepted timestep) with actions fired on the rising
        edge (On Trigger) and optionally on the falling edge (On Release,
        requires One Shot: false). Action lists map an action type to the
        names of the BCs it acts on.
        '''
        if 'Events' not in self.cap_dict.keys():
            return

        rules = []
        for rule_name, rule_dict in self.cap_dict['Events'].items():
            cond_dict = rule_dict['Condition']
            cond_type = cond_dict['Type'].strip().lower()
            if cond_type == 'cell tr':
                if not reac_man:
                    err_str = 'Event "{}": Cell TR condition requires the Reactions block.'.format(rule_name)
                    raise ValueError(err_str)
                cell_index = cond_dict['Cell Index']
                if not 0 <= cell_index < reac_man.n_cells:
                    err_str = 'Event "{}": Cell Index {} out of range for {} reaction cells.'.format(
                        rule_name, cell_index, reac_man.n_cells)
                    raise ValueError(err_str)
                condition = events.tr_cell_condition(cell_index)
            elif cond_type == 'temperature':
                t_loc = cond_dict['T Location']
                if t_loc == 0:
                    node_indices = [0]
                elif t_loc == len(grid_man.mint_list):
                    node_indices = [grid_man.n_tot - 1]
                else:
                    l_ind = grid_man.mint_list[t_loc - 1]
                    node_indices = [l_ind, l_ind + 1]
                condition = events.temperature_condition(node_indices, cond_dict['T Cutoff'])
            else:
                err_str = 'Event "{}": unrecognized condition type {}.'.format(rule_name, cond_dict['Type'])
                raise ValueError(err_str)

            trigger_actions = self.make_event_actions(rule_dict['On Trigger'], bc_man, rule_name)
            release_actions = self.make_event_actions(rule_dict.get('On Release', {}), bc_man, rule_name)
            one_shot = rule_dict.get('One Shot', True)
            if one_shot and release_actions:
                err_str = 'Event "{}": On Release requires One Shot: false.'.format(rule_name)
                raise ValueError(err_str)

            # BCs an event will activate must start inactive
            for action in trigger_actions:
                if isinstance(action, events.activate_bc_action):
                    action.bc.active = False

            rules.append(events.event_rule(condition, trigger_actions, release_actions, one_shot))
        bc_man.event_man = events.event_manager(rules)


    def make_event_actions(self, action_dict, bc_man, rule_name):
        action_classes = {
            'deactivate bc': events.deactivate_bc_action,
            'activate bc': events.activate_bc_action}
        actions = []
        for action_type, bc_names in action_dict.items():
            class_ = action_classes.get(action_type.strip().lower())
            if class_ is None:
                err_str = 'Event "{}": unrecognized action type {}.'.format(rule_name, action_type)
                raise ValueError(err_str)
            if isinstance(bc_names, str):
                bc_names = [bc_names]
            for bc_name in bc_names:
                actions.append(class_(bc_man.get_by_name(bc_name)))
        return actions


    def load_table(self, grid_man):
        '''Load domain information

        Args:
            grid_man (object): grid manager
        '''
        tab_dict = self.cap_dict['Domain Table']

        # Check that each list has the same number of entries
        layer_names = tab_dict['Material Name']
        n_layers = len(layer_names)
        for key in tab_dict:
            if 'Contact Resistance' in key:
                n_mod = n_layers - 1
            else:
                n_mod = 1.*n_layers
            if n_mod != len(tab_dict[key]):
                err_str = 'Incorrect number of entries on {} line'.format(key)
                raise ValueError(err_str)

        # Set the table values on the grid manager
        grid_man.set_table(tab_dict)


    def load_materials(self, mat_man, grid_man):
        '''Parse material info from the input file.

        Args:
            mat_man (object): material manager object
        '''
        mat_dict = self.cap_dict['Materials']
        for a_mat in mat_dict:
            # Make material with prop list
            fv_mat = material.fv_material(a_mat)
            fv_mat.set_rho(mat_dict[a_mat]['rho'])
            fv_mat.set_cp(mat_dict[a_mat]['cp'])
            fv_mat.set_k(mat_dict[a_mat]['k'])
            fv_mat.calc_alpha()

            # Add material to material manager
            mat_man.add_material(fv_mat, a_mat)

        if 'Contact Resistance' in self.cap_dict['Domain Table'].keys():
            mat_man.cont_res = np.asarray(self.cap_dict['Domain Table']['Contact Resistance'])
        else:
            mat_man.cont_res = np.zeros(grid_man.n_layers-1)
        mat_man.add_mesh(grid_man)


    def load_time(self, grid_man):
        '''Parse timing properties

        Args:
            grid_man (object): grid manager

        Returns:
            time_dict (dictionary): timing options
        '''
        time_dict = self.cap_dict['Time']

        # Adaptive (default) or fixed time
        if 'dt' in time_dict.keys():
            time_dict['Fixed Step'] = True
        else:
            time_dict['Fixed Step'] = False
            time_dict['dt'] = 0.0

        # Determine tranisent run
        if time_dict['Run Time'] < 1e-16:
            time_dict['Solution Mode'] = 'Steady'
        else:
            time_dict['Solution Mode'] = 'Transient'

        # Set max steps if not provided
        if 'Max Steps' not in time_dict.keys():
            time_dict['Max Steps'] = 1e7

        # Set time stepper target error if not provided
        if 'Target Error' not in time_dict.keys():
            time_dict['Target Error'] = 1e-7
        else:
            time_dict['Target Error'] = float(time_dict['Target Error'])

        # Set Jacobian lag if not provided
        if 'Maximum Steps Per Jacobian' not in time_dict.keys():
            time_dict['Maximum Steps Per Jacobian'] = 20
        else:
            time_dict['Maximum Steps Per Jacobian'] = int(time_dict['Maximum Steps Per Jacobian'])

        # Set output frequency if not provided
        if 'Output Frequency' not in time_dict.keys():
            time_dict['Output Frequency'] = 1

        # Set print progress if not provided
        if 'Print Progress' not in time_dict.keys():
            time_dict['Print Progress'] = 1

        # Set TR threshold if not provided
        # Units K/s
        if 'TR T Rate Threshold' not in time_dict.keys():
            time_dict['TR T Rate Threshold'] = 100.0
        else:
            time_dict['TR T Rate Threshold'] = float(time_dict['TR T Rate Threshold'])

        # Set initial temperature
        if type(time_dict['T Initial']) is list:
            if len(time_dict['T Initial']) != grid_man.n_layers:
                err_str = 'Number of initial temperatures does not match number of layers.'
                raise ValueError(err_str)
            temp_init = np.zeros(grid_man.n_tot)
            n_start = 0
            for m in range(grid_man.n_layers):
                n_end = grid_man.mint_list[m] + 1
                temp_init[n_start:n_end] = time_dict['T Initial'][m]
                n_start = 1*n_end
            time_dict['T Initial'] = temp_init
        else:
            time_dict['T Initial'] = np.zeros(grid_man.n_tot) + time_dict['T Initial']

        return time_dict
