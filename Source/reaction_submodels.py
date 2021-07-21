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


reaction_submodel_dictionary = {
    'Electrolyte Limiter': 'electrolyte_limiter',
    'Damkohler': 'damkohler_limiter'
}


class rxn_submodel:
    def __init__(self, parent_rxn):
        self.rxn_info = parent_rxn.rxn_info
        self.material_info = parent_rxn.material_info
        self.molecular_weights = parent_rxn.molecular_weights
        self.species_name_list = parent_rxn.species_name_list
        self.n_species = len(self.species_name_list)
        self.small_number = 1.0e-15
        self.name_map = parent_rxn.name_map

        self.setup()


    def setup(self):
        return 0


    def concentration_function(self, my_v):
        return 1


    def concentration_derivative(self, my_v):
        my_dr_part_col = np.zeros(self.n_species)
        return my_dr_part_col


    def evaluate_rate_constant(self, my_v):
        return 1.0


    def evaluate_rate_constant_derivative(self, my_v, my_k):
        return my_k*self.evaluate_rate_constant_derivative_part(my_v)


    def evaluate_rate_constant_derivative_part(self, my_v):
        return 0.0


class electrolyte_limiter(rxn_submodel):
    def setup(self):
        self.species_name = self.rxn_info['Electrolyte Limiter']['Species']
        self.sp_ind = self.name_map[self.species_name]
        self.sp_50 = self.rxn_info['Electrolyte Limiter']['Limiting Constant']


    def concentration_function(self, my_v):
        rho_sp = my_v[self.sp_ind]
        return rho_sp/(self.sp_50 + rho_sp)


    def concentration_derivative(self, my_v):
        my_dr_part_col = np.zeros(self.n_species)
        my_dr_part_col[self.sp_ind] = self.sp_50/(self.sp_50 + my_v[self.sp_ind])**2
        return my_dr_part_col


class damkohler_limiter(rxn_submodel):
    def setup(self):
        dam_info = self.rxn_info['Damkohler']
        T_ref_d = 25. + 273.15
        r_i = dam_info['r_i']
        r_o = dam_info['r_o']
        if 'a_edges' not in self.rxn_info.keys():
            err_str = 'Edge area not found for Damkohler model.'
            raise ValueError(err_str)
        else:
            a_edges = self.rxn_info['a_edges']

        # Combine rate constant and diffusion coefficient to form the Damkohler number
        C_D = (r_o - r_i)*r_o/(r_i*a_edges*self.material_info['rho'])
        A_D_part = dam_info['D']*np.exp(dam_info['E']/(self.rxn_info['R']*T_ref_d))
        self.AD = C_D*float(dam_info['A'])/A_D_part
        self.EDoR = (float(self.rxn_info['E']) - float(dam_info['E']))/float(self.rxn_info['R'])


    def evaluate_rate_constant(self, my_v):
        return 1.0/(1 + self.AD*np.exp(-self.EDoR/my_v[-1]))


    def evaluate_rate_constant_derivative(self, my_v, my_k):
        return my_k**2*self.evaluate_rate_constant_derivative_part(my_v)


    def evaluate_rate_constant_derivative_part(self, my_v):
        return (-1.*self.EDoR/my_v[-1]**2)*self.AD*np.exp(-self.EDoR/my_v[-1])
