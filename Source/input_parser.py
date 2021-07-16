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
import os, sys
import pandas as pd
import yaml
import material
import boundary
import grid
import reaction
import data


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
            print(key)
            print(self.cap_dict[key])


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

        # Materials
        mat_man = material.material_manager()
        self.load_materials(mat_man, grid_man)

        # Time
        time_opts = self.load_time(grid_man)

        # Boundaries
        bc_man = boundary.bc_manager(grid_man)
        self.load_bc(bc_man)

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

        # Data manager
        data_man = data.data_manager(grid_man, reac_man, time_opts, self.cap_dict, self.fold_name, self.file_name)

        return mat_man, grid_man, bc_man, reac_man, data_man, time_opts


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


    def load_bc(self, bc_man):
        '''Parse boundary properties

        Args:
            bc_man (object): boundary condition manager
        '''
        bnd_dict = self.cap_dict['Boundary']

        # Set up BCs
        bc_man.setup(bnd_dict)

        # Set perimeter to cross-sectional area ratio
        oth_dict = self.cap_dict['Other']
        L_y = oth_dict['Y Dimension']
        L_z = oth_dict['Z Dimension']
        bc_man.PA_r = 2.*(L_y + L_z)/(L_y*L_z)


    def load_time(self, grid_man):
        '''Parse timing properties

        Args:
            grid_man (object): grid manager

        Returns:
            time_dict (dictionary): timing options
        '''
        time_dict = self.cap_dict['Time']

        # Determine tranisent run
        if time_dict['Run Time'] < 1e-16:
            time_dict['Solution Mode'] = 'Steady'
            time_dict['dt'] = 0.0
        else:
            time_dict['Solution Mode'] = 'Transient'
            if 'Force Split' in time_dict.keys():
                time_dict['Solution Mode'] += ' Split'

        # Set max steps if not provided
        if 'Max Steps' not in time_dict.keys():
            time_dict['Max Steps'] = 1e7
        
        # Set accuracy order if not provided
        if 'Order' not in time_dict.keys():
            time_dict['Order'] = 1

        # Set output frequency if not provided
        if 'Output Frequency' not in time_dict.keys():
            time_dict['Output Frequency'] = 1

        # Set print progress if not provided
        if 'Print Progress' not in time_dict.keys():
            time_dict['Print Progress'] = 1

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
