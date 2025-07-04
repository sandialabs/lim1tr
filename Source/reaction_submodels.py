########################################################################################
#                                                                                      #
#  Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).  #
#  Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains    #
#  certain rights in this software.                                                    #
#                                                                                      #
#  This software is released under the license detailed in the file, LICENSE.          #
#                                                                                      #
########################################################################################

import numpy as np


reaction_submodel_dictionary = {
    'Electrolyte Limiter': 'electrolyte_limiter',
    'Damkohler': 'damkohler_limiter',
    'Damkohler ri': 'damkohler_limiter_ri'
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


    def concentration_function(self, species_mat):
        return np.ones(species_mat.shape[1])


    def concentration_derivative(self, species_mat):
        dr_ds_part = np.zeros(species_mat.shape)
        return dr_ds_part


    def evaluate_rate_constant(self, T_arr):
        return np.ones(T_arr.shape[0])


    def evaluate_rate_constant_derivative(self, T_arr, my_k):
        return my_k*self.evaluate_rate_constant_derivative_part(T_arr)


    def evaluate_rate_constant_derivative_part(self, T_arr):
        return 0.0


class electrolyte_limiter(rxn_submodel):
    def setup(self):
        self.species_name = self.rxn_info['Electrolyte Limiter']['Species']
        self.sp_ind = self.name_map[self.species_name]
        self.sp_50 = self.rxn_info['Electrolyte Limiter']['Limiting Constant']


    def concentration_function(self, species_mat):
        return species_mat[self.sp_ind,:]/(self.sp_50 + species_mat[self.sp_ind,:])


    def concentration_derivative(self, species_mat):
        dr_ds_part = np.zeros(species_mat.shape)
        dr_ds_part[self.sp_ind,:] = self.sp_50/(self.sp_50 + species_mat[self.sp_ind,:])**2
        return dr_ds_part


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

        # Check for solid particle density
        if 'rho' in dam_info.keys():
            rho_s = dam_info['rho']
        else:
            rho_s = self.material_info['rho']

        # Combine rate constant and diffusion coefficient to form the Damkohler number
        C_D = (r_o - r_i)*r_o/(r_i*a_edges*rho_s)
        A_D_part = dam_info['D']*np.exp(dam_info['E']/(self.rxn_info['R']*T_ref_d))
        self.AD = C_D*float(dam_info['A'])/A_D_part
        self.EDoR = (float(self.rxn_info['E']) - float(dam_info['E']))/float(self.rxn_info['R'])


    def evaluate_rate_constant(self, T_arr):
        return 1.0/(1 + self.AD*np.exp(-self.EDoR/T_arr))


    def evaluate_rate_constant_derivative_part(self, T_arr):
        Da = self.AD*np.exp(-self.EDoR/T_arr)
        return (-1*self.EDoR/T_arr**2)*Da/(1 + Da)


class damkohler_limiter_ri(rxn_submodel):
    def setup(self):
        dam_info = self.rxn_info['Damkohler ri']
        T_ref_d = 25. + 273.15
        r_o = dam_info['r_o']
        if 'a_edges' not in self.rxn_info.keys():
            err_str = 'Edge area not found for Damkohler model.'
            raise ValueError(err_str)
        else:
            a_edges = self.rxn_info['a_edges']

        self.species_name = dam_info['Species']
        self.sp_ind = self.name_map[self.species_name]
        sp_o = self.material_info['Initial Mass Fraction'][self.species_name]*self.material_info['rho']
        self.sp_o_13 = sp_o**(1/3)
        self.sp_13 = sp_o**(1/3)

        # Check for solid particle density
        if 'rho' in dam_info.keys():
            rho_s = dam_info['rho']
        else:
            rho_s = self.material_info['rho']

        # Combine rate constant and diffusion coefficient to form the Damkohler number
        A_D_part = dam_info['D']*np.exp(dam_info['E']/(self.rxn_info['R']*T_ref_d))
        self.AD = r_o*float(dam_info['A'])/(a_edges*rho_s*A_D_part)
        self.EDoR = (float(self.rxn_info['E']) - float(dam_info['E']))/float(self.rxn_info['R'])


    def concentration_function(self, species_mat):
        self.sp_13 = np.maximum(species_mat[self.sp_ind,:],0)**(1/3)
        return np.ones(species_mat.shape[1])


    def evaluate_rate_constant(self, T_arr):
        return self.sp_13/(self.sp_13 + self.AD*np.exp(-self.EDoR/T_arr)*(self.sp_o_13 - self.sp_13))


    def evaluate_rate_constant_derivative_part(self, T_arr):
        Da = self.AD*(self.sp_o_13 - self.sp_13)*np.exp(-self.EDoR/T_arr)
        return (-1*self.EDoR/T_arr**2)*Da/(self.sp_13 + Da)


    def concentration_derivative(self, species_mat):
        # Unimplemented
        dr_ds_part = np.zeros(species_mat.shape)
        return dr_ds_part
