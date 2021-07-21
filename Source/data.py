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
import pickle as p
import pandas as pd
import os


class data_manager:
    def __init__(self, grid_man, reac_man, time_opts, cap_dict, fold_name, file_name):
        '''The basic iteration of the data manager should just
        have a path for saving a pickle/csv of temperature and
        density at every time step in the same folder as the 
        input file.
        '''
        self.cap_dict = cap_dict
        self.fold_name = fold_name
        self.file_name = file_name
        self.data_dict = {}
        self.data_dict['Time'] = [0.0]
        self.data_dict['Grid'] = grid_man.x_node
        self.rate_dict = {}
        self.n_tot = grid_man.n_tot
        self.mint_list = grid_man.mint_list
        self.data_dict['Temperature'] = np.array(np.zeros(self.n_tot) + time_opts['T Initial'], ndmin=2)
        self.data_dict['Interface Temperature'] = self.get_interface_temperatures(self.data_dict['Temperature'])
        self.reac_present = False
        if reac_man:
            self.rate_dict['Time'] = [0.0]
            self.reac_present = True
            self.species_name_list = reac_man.species_name_list
            for spec_name in self.species_name_list:
                self.data_dict[spec_name] = np.array(reac_man.species_density[spec_name], ndmin=2)
                self.rate_dict[spec_name] = np.array(reac_man.species_rate[spec_name], ndmin=2)
            self.rate_dict['HRR'] = np.array(reac_man.heat_release_rate, ndmin=2)
            self.rate_dict['Reaction Temperature Rate'] = np.array(reac_man.temperature_rate, ndmin=2)
        self.data_len = 1
        num_outputs = self.n_tot + self.data_dict['Interface Temperature'].shape[1]
        if reac_man:
            num_outputs += 2*self.n_tot*(len(self.species_name_list) + 1)
        max_entries = 200000
        self.max_len = max(10, int(max_entries/num_outputs))
        self.out_num = 0
        self.output_frequency = time_opts['Output Frequency']


    def get_interface_temperatures(self, T_arr):
        T_interface = np.array(np.zeros(len(self.mint_list) - 1), ndmin=2)
        for m in range(len(self.mint_list) - 1):
            tmp_ind = self.mint_list[m]
            T_interface[0,m] = 0.5*(T_arr[0,tmp_ind] + T_arr[0,tmp_ind + 1])
        return T_interface


    def save_data(self, t_int, reac_man, process_rates=True):
        if ((t_int.n_step - 1)%self.output_frequency != 0):
            return 0

        # Check to see if we need to write a chunk
        if self.data_len >= self.max_len:
            self.write_data()
            self.out_num += 1

            # Reset data dictionary with most recent step
            self.data_dict['Time'] = [t_int.tot_time]
            self.data_dict['Temperature'] = np.array(t_int.T_m1, ndmin=2)
            self.data_dict['Interface Temperature'] = self.get_interface_temperatures(self.data_dict['Temperature'])
            if reac_man:
                self.rate_dict['Time'] = [t_int.tot_time]
                for spec_name in self.species_name_list:
                    self.data_dict[spec_name] = np.array(reac_man.species_density[spec_name], ndmin=2)
                    self.rate_dict[spec_name] = np.array(reac_man.species_rate[spec_name], ndmin=2)
                self.rate_dict['HRR'] = np.array(reac_man.heat_release_rate, ndmin=2)
                self.rate_dict['Reaction Temperature Rate'] = np.array(reac_man.temperature_rate, ndmin=2)
            self.data_len = 1
        else:
            # Append most recent step
            self.data_dict['Time'].append(t_int.tot_time)
            self.data_dict['Temperature'] = np.concatenate((self.data_dict['Temperature'], np.array(t_int.T_m1, ndmin=2)), axis=0)
            interface_temp = self.get_interface_temperatures(np.array(t_int.T_m1, ndmin=2))
            self.data_dict['Interface Temperature'] = np.concatenate((self.data_dict['Interface Temperature'], interface_temp), axis=0)
            if reac_man:
                self.rate_dict['Time'].append(t_int.tot_time)
                for spec_name in self.species_name_list:
                    self.data_dict[spec_name] = np.concatenate((self.data_dict[spec_name], np.array(reac_man.species_density[spec_name], ndmin=2)), axis=0)
                    self.rate_dict[spec_name] = np.concatenate((self.rate_dict[spec_name], np.array(reac_man.species_rate[spec_name], ndmin=2)), axis=0)
                self.rate_dict['HRR'] = np.concatenate((self.rate_dict['HRR'], np.array(reac_man.heat_release_rate, ndmin=2)), axis=0)
                self.rate_dict['Reaction Temperature Rate'] = np.concatenate((self.rate_dict['Reaction Temperature Rate'], np.array(reac_man.temperature_rate, ndmin=2)), axis=0)
            self.data_len += 1


    def write_data(self):
        # Write data to a pickle
        self.data_dict['Time'] = np.asarray(self.data_dict['Time'])
        if self.reac_present:
            self.rate_dict['Time'] = np.asarray(self.rate_dict['Time'])
        tmp_name = self.file_name + '_output_' + str(self.out_num).rjust(4, '0') + '.p'
        output_file = os.path.join(self.fold_name, tmp_name)
        with open(output_file, 'wb') as f:
            p.dump([self.cap_dict, self.data_dict, self.rate_dict], f)


    def compile_data(self):
        # Open first file
        tmp_name = self.file_name + '_output_' + str(0).rjust(4, '0') + '.p'
        output_file = os.path.join(self.fold_name, tmp_name)
        with open(output_file , 'rb') as f:
            my_cap, my_data, my_rate = p.load(f)

        # Open and append remaining files
        for i in range(1,self.out_num + 1):
            tmp_name = self.file_name + '_output_' + str(i).rjust(4, '0') + '.p'
            output_file = os.path.join(self.fold_name, tmp_name)
            with open(output_file , 'rb') as f:
                tmp_cap, tmp_data, tmp_rate = p.load(f)
            my_data['Time'] = np.concatenate((my_data['Time'], tmp_data['Time']), axis=0)
            my_data['Temperature'] = np.concatenate((my_data['Temperature'], tmp_data['Temperature']), axis=0)
            my_data['Interface Temperature'] = np.concatenate((my_data['Interface Temperature'], tmp_data['Interface Temperature']), axis=0)
            if self.reac_present:
                my_rate['Time'] = np.concatenate((my_rate['Time'], tmp_rate['Time']), axis=0)
                for spec_name in self.species_name_list:
                    my_data[spec_name] = np.concatenate((my_data[spec_name], tmp_data[spec_name]), axis=0)
                    my_rate[spec_name] = np.concatenate((my_rate[spec_name], tmp_rate[spec_name]), axis=0)
                my_rate['HRR'] = np.concatenate((my_rate['HRR'], tmp_rate['HRR']), axis=0)
                my_rate['Reaction Temperature Rate'] = np.concatenate((my_rate['Reaction Temperature Rate'], tmp_rate['Reaction Temperature Rate']), axis=0)

        # Write data to a pickle
        tmp_name = self.file_name + '_output.p'
        output_file = os.path.join(self.fold_name, tmp_name)
        with open(output_file ,'wb') as f:
            p.dump([my_cap, my_data, my_rate], f)
