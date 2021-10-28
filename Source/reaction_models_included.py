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
import reaction_model_base


rxn_model_dictionary_included = {
    'Basic': 'basic_rxn',
    'Short': 'simple_short',
    'Zcrit': 'zcrit'
}

class basic_rxn(reaction_model_base.rxn_model):
    def setup(self):
        key_list, val_arr = self.build_species_map('Orders')
        self.orders = np.zeros(self.n_species)
        self.orders[key_list] = val_arr


    def concentration_function(self, my_v):
        '''Basic concentration function.
        Product of rho^mu
        '''
        return np.prod(np.maximum(my_v[:-1],0.0)**self.orders)


    def concentration_derivative(self, my_v):
        '''Derivative of basic concentration function
        w.r.t. each species.
        '''
        my_dr_part_col = np.zeros(self.n_species)
        for kk in range(self.n_species):
            if (abs(self.orders[kk]) > self.small_number):
                order_temp = np.copy(self.orders)
                order_temp[kk] -= 1.
                my_dr_part_col[kk] = self.orders[kk]*np.prod(np.maximum(my_v[:-1],1e-8)**order_temp)
        return my_dr_part_col


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


    def concentration_function(self, my_v):
        conc_func = 1.0
        for reactant in self.reactants:
            if my_v[self.name_map[reactant]] < self.small_number:
                conc_func *= 0.0
            else:
                conc_func *= my_v[self.name_map[reactant]]/(self.short_lim + my_v[self.name_map[reactant]])
        return conc_func


    def concentration_derivative(self, my_v):
        my_dr_part_col = np.zeros(self.n_species)
        for reactant in self.reactants:
            dr_dv = self.short_lim/(self.name_map[reactant] + self.short_lim)**2
            for other_reactant in self.reactants:
                if reactant not in other_reactant:
                    v_conc = my_v[self.name_map[other_reactant]]
                    dr_dv *= v_conc/(self.short_lim + v_conc)
            my_dr_part_col[self.name_map[reactant]] = dr_dv
        return my_dr_part_col


class zcrit(reaction_model_base.rxn_model):
    def setup(self):
        self.rho = self.material_info['rho']
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
        self.z_c = (2*6*12.011)/(self.molecular_weights['Li2CO3']*self.rho*Y_Graphite*BET_C6**0.5)


    def concentration_function(self, my_v):
        '''Critical thickness anode model
        '''
        if self.z_c*my_v[self.name_map['Li2CO3']] >= self.tau_crit:
            crit_fun = np.exp(-self.C_t*self.tau_crit)
        else:
            crit_fun = np.exp(-self.C_t*self.z_c*my_v[self.name_map['Li2CO3']])
        return self.a_e_crit*my_v[self.name_map['C6Li']]*crit_fun


    def concentration_derivative(self, my_v):
        '''Derivative of critical thickness anode model
        w.r.t. each species
        '''
        my_dr_part_col = np.zeros(self.n_species)

        # Zcrit limiter and derivative of Li2CO3
        if self.z_c*my_v[self.name_map['Li2CO3']] >= self.tau_crit:
            crit_fun = np.exp(-self.C_t*self.tau_crit)
        else:
            crit_fun = np.exp(-self.C_t*self.z_c*my_v[self.name_map['Li2CO3']])
            my_dr_part_col[self.name_map['Li2CO3']] = self.a_e_crit*(
                my_v[self.name_map['C6Li']]*crit_fun*(-self.C_t*self.z_c))

        # C6Li derivative
        my_dr_part_col[self.name_map['C6Li']] = self.a_e_crit*crit_fun

        return my_dr_part_col
