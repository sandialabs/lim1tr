########################################################################################
#                                                                                      #
#  Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).  #
#  Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains    #
#  certain rights in this software.                                                    #
#                                                                                      #
#  This software is released under the license detailed in the file, LICENSE.          #
#                                                                                      #
########################################################################################

import boundary_types
import numpy as np


end_boundaries = {
    'adiabatic': 'end_bc',
    'dirichlet': 'end_dirichlet',
    'convection': 'end_convection',
    'heat flux': 'end_flux',
    'radiation': 'end_radiation'
}

end_params = {
    'adiabatic': [],
    'dirichlet': ['T'],
    'convection': ['T', 'h'],
    'heat flux': ['Flux'],
    'radiation': ['T', 'eps']
}

ext_boundaries = {
    'adiabatic': 'ext_bc',
    'convection': 'ext_convection',
    'radiation': 'ext_radiation'
}

ext_params = {
    'adiabatic': [],
    'convection': ['T', 'h'],
    'radiation': ['T', 'eps']
}

required_control_params = {
    'T Cutoff': 'T_cutoff',
    'T Location': 'T_location',
    'T Post': 'T_post',
    'h Post': 'h_post'
}

def factory(location, params, dx_arr, PA_r, mint_list):
    bc_type = params['Type'].strip().lower()
    if location == 'Left' or location == 'Right':
        boundaries = end_boundaries
        required_params = end_params
        check_boundary_type(bc_type, boundaries, location)
        class_ = getattr(boundary_types, boundaries[bc_type])
        parent_bc = class_(dx_arr, location)
    elif location == 'External':
        boundaries = ext_boundaries
        required_params = ext_params
        check_boundary_type(bc_type, boundaries, location)
        class_ = getattr(boundary_types, boundaries[bc_type])
        parent_bc = class_(dx_arr, PA_r)

    temporal_functions = {}
    for param in required_params[bc_type]:
        try:
            param_type = type(params[param])
        except KeyError:
            err_str = f'Parameter {param} not found in {location} {bc_type} boundary.'
            raise KeyError(err_str)

        if param_type is dict:
            initial_value, param_function = parse_temporal_param(params[param])
            temporal_functions[param] = param_function
        elif param_type is float or param_type is int:
            initial_value = params[param]
        else:
            err_str = f'Incorrect input for {param} on {location} boundary.'
            raise ValueError(err_str)

        setattr(parent_bc, param, initial_value)
    parent_bc.setup_params()
    
    if len(temporal_functions) > 0:
        if 'radiation' in parent_bc.name:
            err_str = f'Radiation not supported for transient properties.'
            raise ValueError(err_str)
        wrap_bc = boundary_types.temporal_boundary(parent_bc)
        for param in temporal_functions.keys():
            wrap_bc.add_param(param, temporal_functions[param])
        bc = wrap_bc
    else:
        bc = parent_bc

    if 'Temperature Control' in params.keys() and location != 'External':
        final_bc = boundary_types.end_temperature_control(bc, dx_arr, mint_list)
        control_params = params['Temperature Control']
        for param in required_control_params.keys():  
            if param not in control_params:
                err_str = f'Parameter {param} not found in {location} temperature control boundary.'
                raise KeyError(err_str)
            setattr(final_bc, required_control_params[param], control_params[param])
        final_bc.setup_params()
    elif 'Deactivation Time' in params.keys():
        final_bc = boundary_types.timed_boundary(bc, params['Deactivation Time'])
    else:
        final_bc = bc

    return final_bc


def check_boundary_type(bc_type, boundaries, location):
    if bc_type not in boundaries.keys():
        err_str = f'Bondary type {bc_type} for {location} boundary not found.'
        raise KeyError(err_str)
    

def parse_temporal_param(temporal_param):
    if 'Table' in temporal_param.keys():
        table = np.genfromtxt(temporal_param['Table'], delimiter=',')
        param_function = boundary_types.table_function(table[:,0], table[:,1])
        initial_value = param_function(0.0)
    else:
        initial_value = temporal_param['Initial']
        rate = temporal_param['Rate']
        param_function = boundary_types.ramp_function(rate, initial_value)
    return initial_value, param_function
