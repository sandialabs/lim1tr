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


class rxn_model:
    def __init__(self, rxn_info, reac_man):
        self.rxn_info = rxn_info
        self.molecular_weights = reac_man.molecular_weights
        self.species_name_list = reac_man.species_name_list
        self.n_species = len(self.species_name_list)

        self.rxn_info['Reactants'] = self.convert_to_mass(self.rxn_info['Reactants'])
        self.rxn_info['Products'] = self.convert_to_mass(self.rxn_info['Products'])

        # Make a map from species names to state vector index
        self.name_map = {}
        for i in range(len(self.species_name_list)):
            self.name_map[self.species_name_list[i]] = i

        self.setup(reac_man)


    def setup(self, reac_man):
        return 0.0


    def convert_to_mass(self, my_dict):
        for key, value in my_dict.items():
            my_dict[key] = value*self.molecular_weights[key]

        return my_dict


    def build_species_map(self, info_key):
        key_list = []
        val_list = []
        for key, value in self.rxn_info[info_key].items():
            try:
                # print(key, self.species_name_list.index(key))
                key_list.append(self.species_name_list.index(key))
                val_list.append(value)
            except ValueError:
                err_str = 'The species {} was not found in the species list in the species block.'.format(key)
                raise ValueError(err_str)

        return key_list, np.asarray(val_list)


    def build_reactant_map(self):
        key_list, val_arr = self.build_species_map('Reactants')
        val_arr = val_arr/np.sum(val_arr)

        return key_list, val_arr


    def build_product_map(self):
        key_list, val_arr = self.build_species_map('Products')
        val_arr = val_arr/np.sum(val_arr)

        return key_list, val_arr
