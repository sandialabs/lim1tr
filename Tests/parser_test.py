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
import time
import sys
sys.path.append('../Source')
import input_parser


# Supply file name
file_name = './Inputs/simple_source.yaml'

# Parse file to get material manager, grid manager, boundary manager, reaction manager, and timing options
a_parser = input_parser.input_parser(file_name)
mat_man, grid_man, bc_man, reac_man, data_man, time_opts = a_parser.apply_parse()
