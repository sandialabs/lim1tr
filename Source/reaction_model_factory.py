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
import reaction_models


def rxn_model_factory(rxn_info, material_info):
    ''' Factory for reaction models. If there are no sub models,
    then a single reaction model is returned. If there are sub
    models, a model chain object is returned that evaluates the
    concentration function, rate constant, and their jacobians for
    all models in the reaction.
    '''
    frac_mat_col = np.zeros(len(material_info['Names']))

    # Make the parent model
    class_ = getattr(reaction_models, reaction_models.rxn_model_dictionary[rxn_info['Type']])
    parent_model = class_(rxn_info, material_info)

    # Build reactant map
    key_list, val_arr = parent_model.build_reactant_map()
    frac_mat_col[key_list] -= val_arr

    # Build product map
    key_list, val_arr = parent_model.build_product_map()
    frac_mat_col[key_list] += val_arr

    # Check for sub-models
    my_models = [parent_model]
    for sub_model in reaction_models.reaction_submodel_dictionary.keys():
        if sub_model in rxn_info.keys():
            class_ = getattr(reaction_models, reaction_models.reaction_submodel_dictionary[sub_model])
            my_sub_model = class_(parent_model)
            my_models.append(my_sub_model)

    # Make a product rule evaluator if necessary
    if len(my_models) > 1:
        my_rxn_model = model_chain(my_models)
    else:
        my_rxn_model = parent_model

    return my_rxn_model, frac_mat_col


class model_chain:
    def __init__(self, my_funcs):
        self.my_funcs = my_funcs
        self.n_funs = len(self.my_funcs)

        # Pull key parameters from the parent reactions
        self.n_species = self.my_funcs[0].n_species
        self.H_rxn = self.my_funcs[0].H_rxn


    def concentration_function(self, species_mat):
        conc_func = 1.0
        i = 0
        for func in self.my_funcs:
            conc_func *= func.concentration_function(species_mat)
            i += 1
        return conc_func


    def concentration_derivative(self, species_mat):
        dr_ds_part = np.zeros(species_mat.shape)

        conc_funcs = np.zeros([self.n_funs, species_mat.shape[1]])
        conc_ders = np.zeros([self.n_funs, self.n_species, species_mat.shape[1]])
        for i in range(self.n_funs):
            conc_funcs[i,:] = self.my_funcs[i].concentration_function(species_mat)
            conc_ders[i,:,:] = self.my_funcs[i].concentration_derivative(species_mat)

        conc_prods = np.zeros([self.n_funs, species_mat.shape[1]])
        for i in range(self.n_funs):
            conc_prods[i,:] = np.prod(np.delete(conc_funcs, i, axis=0), axis=0)

        for j in range(self.n_species):
            dr_ds_part[j,:] = np.sum(conc_ders[:,j,:]*conc_prods, axis=0)

        return dr_ds_part


    def evaluate_rate_constant(self, T_arr):
        my_k = 1.0
        i = 0
        for func in self.my_funcs:
            my_k *= func.evaluate_rate_constant(T_arr)
            i += 1
        return my_k


    def evaluate_rate_constant_derivative(self, T_arr, my_k):
        my_k_dT = 0.0
        i = 0
        for func in self.my_funcs:
            my_k_dT += func.evaluate_rate_constant_derivative_part(T_arr)
            i += 1
        return my_k*my_k_dT
