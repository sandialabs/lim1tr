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
import time


class reaction_layer:
    def __init__(self, bounds, offset, n_tot):
        '''setup'''
        self.bounds = bounds
        self.n_tot = n_tot
        self.my_n = bounds[1] - bounds[0]
        self.offset = 1*offset
        self.slice_time = 0


    def set_n_rxn_nodes(self, n_rxn_nodes):
        self.n_rxn_nodes = 1*n_rxn_nodes


    def set_reaction_system(self, reaction_system):
        self.reaction_system = reaction_system
        self.n_species = self.reaction_system.n_species


    def evaluate_rhs(self, t, state):
        # State should already be reshaped
        # Slice state using bounds
        T_arr, species_mat = self.slice_state(state)

        # Evalutate reaction system for those nodes
        # Just pass in a reshaped state, then slice these nodes and feed
        # slice directly
        return self.reaction_system.evaluate_ode(t, T_arr, species_mat)


    def evaluate_jacobian(self, t, state):
        # Slice state using bounds
        T_arr, species_mat = self.slice_state(state)

        # Evalutate reaction system jacobian for those nodes
        return self.reaction_system.evaluate_jacobian(t, T_arr, species_mat)


    def slice_state(self, state):
        # Slice state
        t_st = time.time()
        T_arr = state[self.bounds[0]:self.bounds[1]]

        species_mat = np.zeros([self.n_species, self.my_n])
        for i in range(self.n_species):
            b_1 = self.n_tot + self.offset + i*self.n_rxn_nodes
            b_2 = b_1 + self.my_n
            species_mat[i,:] = state[b_1:b_2]
        self.slice_time += time.time() - t_st
        return T_arr, species_mat
