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
import parallel_reaction_test
import os, sys, time


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
test_status.append(reaction_test.fd_check('jac_test', e_thresh=1e-4, du=1e-2))
test_status.append(reaction_test.fd_check('jac_test_single'))
test_status.append(reaction_test.short_rxn_C6Li())
test_status.append(reaction_test.short_rxn_CoO2())
test_status.append(reaction_test.zcrit_rxn())
test_status.append(reaction_test.fd_check('anode_only', e_thresh=2e-7, du=1e-6))
test_status.append(reaction_test.damkohler_anode_test())
test_status.append(reaction_test.fd_check('damkohler_anode', e_thresh=4e-6, du=1e-6))

# Reaction manager tests
test_status.append(reaction_test.one_unique_system_test())
test_status.append(reaction_test.two_unique_system_test())
test_status.append(reaction_test.three_unique_system_test())
test_status.append(reaction_test.map_system_index_to_node_test())

# Sub reaction tests
test_status.append(reaction_test.concentration_product_rule_test())
test_status.append(reaction_test.rate_constant_product_rule_test_ec())
test_status.append(reaction_test.rate_constant_product_rule_test_exp())

# Parallel reaction tests
if sys.version_info[0] >= 3:
    test_status.append(parallel_reaction_test.parallel_node_map_test())
else:
    print('Python 2 detected, skipping parallel tests...')

print('Complete')
print('Passed {} of {} tests in {:0.3f} seconds.'.format(sum(test_status), len(test_status), time.time() - start_time))
