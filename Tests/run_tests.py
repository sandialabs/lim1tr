########################################################################################
#                                                                                      #
#  Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).  #
#  Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains    #
#  certain rights in this software.                                                    #
#                                                                                      #
#  This software is released under the license detailed in the file, LICENSE.          #
#                                                                                      #
########################################################################################

import unittest
import os, sys, time
from build_sys_test import build_tests
from material_test import material_tests
from steady_cond import steady_cond_tests
from trans_cond import trans_cond_tests
from reaction_test import reaction_tests
from reaction_manager_test import reaction_manager_tests
from sub_reaction_test import sub_reaction_tests
if sys.version_info[0] >= 3:
    from parallel_reaction_test import parallel_reaction_tests
from radiation_test import rad_tests


if __name__ == '__main__':
    # Make figures folder
    if not os.path.exists('Figures'):
        os.mkdir('Figures')

    unittest.main()
