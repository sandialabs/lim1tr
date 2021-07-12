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
    'Short': 'tanh_short',
    'Zcrit': 'zcrit'
}

class basic_rxn(reaction_model_base.rxn_model):
    def setup(self, reac_man):
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


class tanh_short(reaction_model_base.rxn_model):
    def setup(self, reac_man):
        '''Setup: there are several constants we can make to cut down on operations
        in the concentration function evaluation
        '''
        self.conc_scale = self.rxn_info['Concentration Scale']
        self.short_pow = self.rxn_info['Short Power']
        self.t_scale = 50.
        self.t_scale_sc = self.t_scale*self.conc_scale
        self.short_slope = 0.15
        self.inverse_slope_plus_one = 1./(1. + self.short_slope)
        self.short_slope_sc = self.short_slope*self.conc_scale


    def concentration_function(self, my_v):
        '''Short circuit model
        '''
        # Convert to molar concentrations of C6Li and CoO2
        conc_C6Li = max(my_v[self.name_map['C6Li']]/79.007, 1e-10)
        conc_CoO2 = max(my_v[self.name_map['CoO2']]/90.931, 1e-10)
        if conc_C6Li > conc_CoO2:
            # print('CoO2 limiting')
            return (np.tanh(self.t_scale_sc*conc_CoO2) + self.short_slope_sc*conc_CoO2)*self.inverse_slope_plus_one
        else:
            # print('C6Li limiting')
            return (np.tanh(self.t_scale_sc*conc_C6Li) + self.short_slope_sc*conc_C6Li)*self.inverse_slope_plus_one


    def concentration_derivative(self, my_v):
        '''Derivative of short circuit model
        w.r.t. each species
        '''
        my_dr_part_col = np.zeros(self.n_species)

        # Convert to molar concentrations of C6Li and CoO2
        conc_C6Li = max(my_v[self.name_map['C6Li']]/79.007, 1e-10)
        conc_CoO2 = max(my_v[self.name_map['CoO2']]/90.931, 1e-10)
        if conc_C6Li > conc_CoO2:
            aa = self.t_scale_sc*(1. - np.tanh(self.t_scale_sc*conc_CoO2)**2) + self.short_slope_sc
            my_dr_part_col[self.name_map['CoO2']] = aa*self.inverse_slope_plus_one/90.931
        else:
            aa = self.t_scale_sc*(1. - np.tanh(self.t_scale_sc*conc_C6Li)**2) + self.short_slope_sc
            my_dr_part_col[self.name_map['C6Li']] = aa*self.inverse_slope_plus_one/79.007
        return my_dr_part_col


class zcrit(reaction_model_base.rxn_model):
    def setup(self, reac_man):
        self.rho = reac_man.rho
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
