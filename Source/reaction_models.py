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
import os


# Include supplied models
from reaction_models_included import *
rxn_model_dictionary = {}
rxn_model_dictionary.update(rxn_model_dictionary_included)

# Check for user models and add
try:
    from reaction_models_user import *
    rxn_model_dictionary.update(rxn_model_dictionary_user)
except ImportError:
    pass

# Include sub-models
from reaction_submodels import *
