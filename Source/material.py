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


class material_manager:
    def __init__(self):
        '''Handles materials and property evaluation.
        It owns the nodal/interface values of each material property.
        '''
        self.my_materials = []
        self.my_m_names = []
        self.loc_dict = {}
        self.u_mats = 0


    def add_mesh(self, grid_man):
        '''Loads in the mesh information and sets material interface bounds

        Args:
            grid_man (object): grid manager
        '''
        self.dx_arr = grid_man.dx_arr
        self.mat_nodes = grid_man.mat_nodes
        self.layer_names = grid_man.layer_names

        self.n_layers = grid_man.n_layers
        self.n_tot = grid_man.n_tot

        self.rho_arr = np.zeros(self.n_tot)
        self.cp_arr = np.zeros(self.n_tot)
        self.k_arr = np.zeros(self.n_tot - 1)
        self.m_arr = np.zeros(self.n_tot)

        self.interface_ids = grid_man.interface_ids
        self.internal_ids = grid_man.internal_ids


    def add_material(self, fv_mat, m_name):
        '''Add a material

        Args:
            fv_mat (object): fv_material object
            m_name (str): name of the fv_material object
        '''
        if m_name not in self.my_m_names:
            self.my_materials.append(fv_mat)
            self.my_m_names.append(m_name)
            self.loc_dict[m_name] = self.u_mats
            self.u_mats += 1


    def get_material(self, m_name):
        '''Get a material object by name

        Args:
            m_name (str): name of the fv_material object
        '''
        if m_name not in self.loc_dict.keys():
            err_str = 'Material {} was not defined in the input file.'.format(m_name)
            raise ValueError(err_str)
        return self.my_materials[self.loc_dict[m_name]]


    def eval_props(self):
        '''Evaluate properties at nodes and interfaces
        '''
        # Evaluate nodal properties
        for i in range(self.n_tot):
            # Evaluate rho
            self.rho_arr[i] = self.get_material(self.mat_nodes[i]).eval_rho()

            # Evaluate cp
            self.cp_arr[i] = self.get_material(self.mat_nodes[i]).eval_cp()

        # Evaluate the mass matrix
        self.m_arr = self.rho_arr*self.cp_arr*self.dx_arr
        self.i_m_arr = 1/self.m_arr
        self.i_rcp = self.i_m_arr*self.dx_arr

        # Evaluate interface properties
        # Internal interfaces
        for i in self.internal_ids:
            self.k_arr[i] = self.get_material(self.mat_nodes[i]).eval_k()

        # Material interfaces
        m = 0
        for i in self.interface_ids:
            kt_i = self.get_material(self.mat_nodes[i]).eval_k()
            kt_i1 = self.get_material(self.mat_nodes[i+1]).eval_k()
            R_i = 0.5*self.dx_arr[i]/kt_i
            R_i1 = 0.5*self.dx_arr[i+1]/kt_i1
            R_tot = R_i + R_i1 + self.cont_res[m]
            self.k_arr[i] = 0.5*(self.dx_arr[i] + self.dx_arr[i+1])/R_tot
            m += 1


class fv_material:
    def __init__(self,m_name):
        '''Generic material.
        Allows setting and evaluation of properties.
        Eventually this could accomodate temperature
        dependent properties where eval could take a
        temperature argument.
        '''
        self.m_name = m_name


    def set_rho(self, rho_in):
        self.rho = rho_in


    def eval_rho(self):
        return self.rho


    def set_cp(self, cp_in):
        self.cp = cp_in


    def eval_cp(self):
        return self.cp


    def set_k(self, k_in):
        self.k = k_in


    def eval_k(self):
        return self.k


    def calc_alpha(self):
        self.alpha = self.k/(self.rho*self.cp)
