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
        self.reactants = self.rxn_info['Reactants'].keys()
        for key in self.rxn_info['Reactants']:
            total_reactants += self.rxn_info['Reactants'][key]
        self.A = voltage*total_reactants/(short_resistance*charge_kmol*volume)
        self.H_rxn = voltage*charge_kmol/total_reactants


    def concentration_function(self, my_v):
        conc_func = 1.0
        for reactant in self.reactants:
            if my_v[self.name_map[reactant]] < self.small_number:
                conc_func = 0.0
        return conc_func


    def concentration_derivative(self, my_v):
        return np.zeros(self.n_species)


class zcrit(reaction_model_base.rxn_model):
    def setup(self):
        self.rho = self.material_info['rho']
        BET_C6 = self.rxn_info['BET_C6']
        self.tau_crit = self.rxn_info['tau_crit']
        self.C_t = self.rxn_info['C_t']
        Y_Graphite = self.rxn_info['Y_Graphite']

        # Internal parameters (can make these user accessible)
        m_EC_50 = 1./200.
        nEDexp = 1.22
        CED = 0.31

        # Calculated parameters needed for model
        self.aEdges = CED*BET_C6**nEDexp
        self.z_c = (2*6*12.011)/(self.molecular_weights['Li2CO3']*self.rho*Y_Graphite*BET_C6**0.5)
        self.rho_50 = BET_C6*self.rho*Y_Graphite*m_EC_50


    def concentration_function(self, my_v):
        '''Critical thickness anode model
        '''
        rho_fun = my_v[self.name_map['C6Li']]*my_v[self.name_map['EC']]/(self.rho_50 + my_v[self.name_map['EC']])
        if self.z_c*my_v[self.name_map['Li2CO3']] >= self.tau_crit:
            crit_fun = np.exp(-self.C_t*self.tau_crit)
        else:
            crit_fun = np.exp(-self.C_t*self.z_c*my_v[self.name_map['Li2CO3']])
        return self.aEdges*rho_fun*crit_fun


    def concentration_derivative(self, my_v):
        '''Derivative of critical thickness anode model
        w.r.t. each species
        '''
        my_dr_part_col = np.zeros(self.n_species)

        # EC calculations
        ec_frac = my_v[self.name_map['EC']]/(self.rho_50 + my_v[self.name_map['EC']])
        rho_fun = my_v[self.name_map['C6Li']]*ec_frac
        rho_fun_dec = my_v[self.name_map['C6Li']]*self.rho_50/(self.rho_50 + my_v[self.name_map['EC']])**2

        # Zcrit limiter and derivative of Li2CO3
        if self.z_c*my_v[self.name_map['Li2CO3']] >= self.tau_crit:
            crit_fun = np.exp(-self.C_t*self.tau_crit)
        else:
            crit_fun = np.exp(-self.C_t*self.z_c*my_v[self.name_map['Li2CO3']])
            my_dr_part_col[self.name_map['Li2CO3']] = self.aEdges*rho_fun*crit_fun*(-self.C_t*self.z_c)

        # EC derivative
        my_dr_part_col[self.name_map['EC']] = self.aEdges*rho_fun_dec*crit_fun

        # C6Li derivative
        my_dr_part_col[self.name_map['C6Li']] = self.aEdges*ec_frac*crit_fun

        return my_dr_part_col
