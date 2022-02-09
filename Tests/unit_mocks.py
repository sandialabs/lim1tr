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


other_mock = {'Y Dimension': 1.0,
    'Z Dimension': 1.0,
    'Reaction Only': 0,
    'DSC Mode': 0,
    'DSC Rate': 0.0}


time_opts_mock = {'Solution Mode': 'Steady',
    'Print Progress': 1,
    'Print Every N Steps': 10,
    'Number of Cores': 1,
    'Order': 1}


class grid_mock:
    def __init__(self):
        self.layer_names = ['A']
        self.layer_dx = [0.1]
        self.layer_thickness = [1.0]
        self.n_layers = 1
        self.n_tot = 10
        self.mint_list = [9]
        self.first_node_list = [0]
        self.mat_nodes = np.asarray(self.layer_names*self.n_tot)
        self.dx_arr = np.zeros(self.n_tot) + self.layer_dx[0]
        self.x_node = np.zeros(self.n_tot)
        for i in range(self.n_tot):
            self.x_node[i] = 0.5*self.dx_arr[i] + np.sum(self.dx_arr[:i])
        self.PA_r = 0.5


basic_rxn_info_mock = {
    'A': 1.0e+9,
    'E': 100000,
    'R': 8.314,
    'H': -1.0e+6,
    'Reactants':
        {'R1': 1,
        'R2': 1},
    'Products':
        {'P': 1},
    'Orders':
        {'R1': 1}
}

material_info_mock = {
    'Names': ['R1', 'R2', 'P', 'Inert'],
    'Molecular Weights': {
        'R1': 1.0,
        'R2': 1.0,
        'P': 1.0,
        'Inert': 0.0},
    'rho': 2000.0,
    'cp': 800.0
}
