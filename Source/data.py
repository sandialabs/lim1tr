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
import pickle as p
import os


class data_manager:
    def __init__(self, grid_man, reac_man, cap_dict, fold_name, file_name):
        self.cap_dict = cap_dict
        self.fold_name = fold_name
        self.file_name = file_name
        self.data_dict = {}
        self.data_dict['Grid'] = grid_man.x_node
        self.data_dict['Layer Map'] = grid_man.layer_map
        self.rate_dict = {}
        self.n_tot = grid_man.n_tot
        self.mint_list = grid_man.mint_list
        self.reac_present = False
        self.dof_name_list = ['Temperature']
        if reac_man:
            self.reac_present = True
            self.dof_name_list += reac_man.species_name_list


    def format_data(self, t, q):
        # Format data into nice dictionaries
        self.data_dict['Time'] = t
        for i in range(len(self.dof_name_list)):
            self.data_dict[self.dof_name_list[i]] = q[:,i*self.n_tot:(i+1)*self.n_tot]

        T_interface = np.zeros([t.shape[0], len(self.mint_list) - 1])
        for m in range(len(self.mint_list) - 1):
            tmp_ind = self.mint_list[m]
            T_interface[:,m] = 0.5*(q[:, tmp_ind] + q[:, tmp_ind + 1])
        self.data_dict['Interface Temperature'] = T_interface


    def write_data(self):
        # Write data to a pickle
        tmp_name = self.file_name + '_output.p'
        output_file = os.path.join(self.fold_name, tmp_name)
        with open(output_file, 'wb') as f:
            p.dump([self.cap_dict, self.data_dict, self.rate_dict], f)
