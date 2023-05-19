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
import reaction_model_base


rxn_model_dictionary_included = {
    'Basic': 'basic_rxn',
    'Short': 'simple_short',
    'Zcrit': 'zcrit'
}

class basic_rxn(reaction_model_base.rxn_model):
    def setup(self):
        key_list, val_arr = self.build_species_map('Orders')
        self.orders = np.zeros([self.n_species,1])
        self.orders[key_list,0] = val_arr
        self.d_orders = np.zeros([self.n_species, self.n_species])
        self.d_s_inds = []
        for kk in range(self.n_species):
            if (abs(self.orders[kk,0]) > self.small_number):
                self.d_orders[:,kk] = np.copy(self.orders[:,0])
                self.d_orders[kk,kk] -= 1.
                self.d_s_inds.append(kk)


    def concentration_function(self, species_mat):
        '''Basic concentration function.
        Product of rho^mu
        '''
        return np.prod(np.power(np.maximum(species_mat,0.0), self.orders), axis=0)


    def concentration_derivative(self, species_mat):
        '''Derivative of basic concentration function
        w.r.t. each species.
        '''
        dr_ds_part = np.zeros(species_mat.shape)
        for kk in self.d_s_inds:
            dr_ds_part[kk,:] = self.orders[kk,0]*np.prod(
                np.power(np.maximum(species_mat,1e-8), self.d_orders[:,[kk]]), axis=0)
        return dr_ds_part


class simple_short(reaction_model_base.rxn_model):
    def setup(self):
        voltage = self.rxn_info['Voltage']
        short_resistance = self.rxn_info['Short Resistance']
        volume = self.rxn_info['Volume']

        charge_kmol = 1000*6.022e23*1.6023e-19
        total_reactants = 0.0
        self.reactants = list(self.rxn_info['Reactants'].keys())
        for key in self.rxn_info['Reactants']:
            total_reactants += self.rxn_info['Reactants'][key]
        self.A = voltage*total_reactants/(short_resistance*charge_kmol*volume)
        self.H_rxn = voltage*charge_kmol/total_reactants
        self.short_lim = 1e-6


    def concentration_function(self, species_mat):
        conc_func = 1.0
        for reactant in self.reactants:
            conc_func *= species_mat[self.name_map[reactant],:]/(
                self.short_lim + species_mat[self.name_map[reactant],:])
            conc_func *= (species_mat[self.name_map[reactant],:] > self.small_number)
        return conc_func


    def concentration_derivative(self, species_mat):
        dr_ds_part = np.zeros(species_mat.shape)
        for reactant in self.reactants:
            r_slice = species_mat[self.name_map[reactant],:]
            r_slice *= (species_mat[self.name_map[reactant],:] > self.small_number)
            dr_dv = self.short_lim/(r_slice + self.short_lim)**2
            for other_reactant in self.reactants:
                if reactant not in other_reactant:
                    v_conc = species_mat[self.name_map[other_reactant],:]
                    dr_dv *= v_conc/(self.short_lim + v_conc)
            dr_ds_part[self.name_map[reactant],:] = dr_dv
        return dr_ds_part


class zcrit(reaction_model_base.rxn_model):
    def setup(self):
        rho = self.material_info['rho']
        BET_C6 = self.rxn_info['BET_C6']
        self.tau_crit = self.rxn_info['tau_crit']
        self.C_t = self.rxn_info['C_t']
        Y_Graphite = self.rxn_info['Y_Graphite']

        # Internal parameters (can make these user accessible)
        nEDexp = 1.22
        CED = 0.31

        # Calculated parameters needed for model
        self.a_e_crit = CED*BET_C6**nEDexp
        self.rxn_info['a_edges'] = self.a_e_crit*1000  # specific edge area in m2/kg
        self.z_c = (2*6*12.011)/(self.molecular_weights['Li2CO3']*rho*Y_Graphite*BET_C6**0.5)


    def concentration_function(self, species_mat):
        '''Critical thickness anode model
        '''
        tau = self.z_c*species_mat[self.name_map['Li2CO3'],:]
        tau[tau >= self.tau_crit] = self.tau_crit
        con_fun = 0
        try:
            con_fun = self.a_e_crit*species_mat[self.name_map['C6Li'],:]*np.exp(-self.C_t*tau)
            con_fun = np.asarray_chkfinite(con_fun)
        except ValueError:
            print(species_mat[self.name_map['Li2CO3'],:])
            print(species_mat[self.name_map['C6Li'],:])
        return con_fun

    def concentration_derivative(self, species_mat):
        '''Derivative of critical thickness anode model
        w.r.t. each species
        '''
        dr_ds_part = np.zeros(species_mat.shape)
        tau = self.z_c*species_mat[self.name_map['Li2CO3'],:]
        tau[tau >= self.tau_crit] = self.tau_crit
        crit_fun = self.a_e_crit*np.exp(-self.C_t*tau)

        # Li2CO3 derivative
        dr_dsalt_part = species_mat[self.name_map['Li2CO3'],:]*crit_fun*(-self.C_t*self.z_c)
        dr_dsalt_part[tau >= self.tau_crit] = 0.0
        dr_ds_part[self.name_map['Li2CO3'],:] = dr_dsalt_part

        # C6Li derivative
        dr_ds_part[self.name_map['C6Li'],:] = crit_fun

        return dr_ds_part
