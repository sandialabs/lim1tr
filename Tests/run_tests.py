########################################################################################
#                                                                                      #
#  Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).  #
#  Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains    #
#  certain rights in this software.                                                    #
#                                                                                      #
#  This software is released under the license detailed in the file, LICENSE.          #
#                                                                                      #
########################################################################################

import build_sys_test
import material_test
import steady_cond
import trans_cond
import reaction_test
import os, time


# Make figures folder
if not os.path.exists('Figures'):
    os.mkdir('Figures')

test_status = []
start_time = time.time()

# Build tests
test_status.append(build_sys_test.cond_apply_test())
test_status.append(build_sys_test.bc_apply_test())

# Material tests
test_status.append(material_test.k_interface())

# Steady conduction tests
test_status.append(steady_cond.simple_steady_cond())
test_status.append(steady_cond.end_conv_steady_cond())
test_status.append(steady_cond.end_conv_steady_cond_stack())
test_status.append(steady_cond.exterior_steady_cond())
test_status.append(steady_cond.contact_resistance())
test_status.append(steady_cond.left_flux_right_conv())
test_status.append(steady_cond.left_conv_right_flux())

# Transient conduction test
test_status.append(trans_cond.trans_end_conv_bdf1())
test_status.append(trans_cond.trans_end_conv_cn())
test_status.append(trans_cond.trans_end_conv_bdf1_split())
test_status.append(trans_cond.trans_end_conv_cn_split())
test_status.append(trans_cond.trans_ext_conv_bdf1())
test_status.append(trans_cond.trans_ext_conv_cn())
test_status.append(trans_cond.trans_ext_conv_bdf1_split())
test_status.append(trans_cond.trans_ext_conv_cn_split())
test_status.append(trans_cond.trans_end_flux_cn())
test_status.append(trans_cond.deactivate_bcs_test())

# Single reaction tests
test_status.append(reaction_test.single_rxn_temperature_ramp())
test_status.append(reaction_test.fd_check('jac_test'))
test_status.append(reaction_test.fd_check('jac_test_single'))
test_status.append(reaction_test.short_rxn())
test_status.append(reaction_test.short_rxn_v2())
test_status.append(reaction_test.zcrit_rxn())

# Reaction manager tests
test_status.append(reaction_test.one_unique_system_test())
test_status.append(reaction_test.two_unique_system_test())
test_status.append(reaction_test.three_unique_system_test())
test_status.append(reaction_test.map_system_index_to_node_test())

print('Complete')
print('Passed {} of {} tests in {:0.3f} seconds.'.format(sum(test_status), len(test_status), time.time() - start_time))
