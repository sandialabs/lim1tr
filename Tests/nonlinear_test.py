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
import scipy as sp
import time, sys, os
sys.path.append('../')
import main_fv



def trans_end_conv():
    file_name = os.getcwd() + '/Inputs/trans_end_conv_bdf1.yaml'

    # Run model
    model = main_fv.lim1tr_model(file_name)
    model.parser.cap_dict['Time']['Run Time'] = model.parser.cap_dict['Time']['dt']
    eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()


if __name__ == '__main__':
    trans_end_conv()