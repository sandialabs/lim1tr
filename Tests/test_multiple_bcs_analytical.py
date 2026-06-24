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
import numpy as np
import sys
import os
import tempfile
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add the Source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../Source'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
import main_fv


class TestMultipleBCsAnalytical(unittest.TestCase):
    def setUp(self):
        self.plotting = True


    def convection_analytical(self, T_initial, T_inf, h, A, m, c, t):
        '''Analytical solution for convection on lumped thermal mass

        Solves: m*c*dT/dt = -h*A*(T - T_inf)
        Solution: T(t) = T_inf + (T_initial - T_inf) * exp(-t/tau)
        where tau = m*c/(h*A) is the thermal time constant
        '''
        tau = m * c / (h * A)
        return T_inf + (T_initial - T_inf) * np.exp(-t / tau)


    def flux_analytical(self, T_initial, q_flux, A_cross, m, c, t):
        '''Analytical solution for constant heat flux boundary condition

        Solves: m*c*dT/dt = q_flux * A_cross
        Solution: T(t) = T_initial + (q_flux * A_cross / (m*c)) * t
        '''
        return T_initial + (q_flux * A_cross / (m * c)) * t


    def combined_analytical(self, T_initial, T_inf, h, A_conv, A_flux, q_flux, m, c, t):
        '''Approximate analytical solution for combined convection and radiation

        Combines both heat transfer mechanisms using linearized radiation.

        Solves: m*c*dT/dt = -h*A_conv*(T - T_inf) + q_flux*A_flux
        Solution: T(t) = T_inf + (T_initial - T_inf) * [exp(-at) + ((b/a)/(T_initial - T_inf))*(1 - exp(-at))]
        where a = h*A_conv/m*c and b = q_flux*A_flux/m*c
        '''
        a = h*A_conv/(m*c)
        b = q_flux*A_flux/(m*c)
        return T_inf + (T_initial - T_inf) * (np.exp(-a*t) + ((b/a)/(T_initial - T_inf))*(1 - np.exp(-a*t)))


    def calculate_errors(self, T_numerical, T_analytical):
        '''Calculate various error metrics for validation'''
        errors = T_numerical - T_analytical

        return {
            'mse': np.mean(errors**2),
            'rmse': np.sqrt(np.mean(errors**2)),
            'max_error': np.max(np.abs(errors)),
            'relative_error': np.max(np.abs(errors) / np.abs(T_analytical + 1e-10)),  # Avoid division by zero
            'mean_absolute_error': np.mean(np.abs(errors))
        }


    def plot_results(self, t_values, T_numerical, T_analytical, case_name, error_metrics):
        '''Plot numerical vs analytical results and save as PNG file'''
        if not self.plotting:
            return

        # Create figure
        plt.figure(figsize=(12, 6))

        # Plot temperature vs time
        plt.subplot(1, 2, 1)
        plt.plot(t_values, T_numerical, 'o-', label='Numerical', markersize=4)
        plt.plot(t_values, T_analytical, '-', label='Analytical', linewidth=2)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Temperature (K)', fontsize=12)
        plt.title(f'{case_name} - Temperature vs Time', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # Plot error vs time
        plt.subplot(1, 2, 2)
        errors = T_numerical - T_analytical
        plt.plot(t_values, errors, 'o-', markersize=3, color='red')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Error (K)', fontsize=12)
        plt.title(f'{case_name} - Error vs Time\nRMSE: {error_metrics["rmse"]:.2e} K, Max: {error_metrics["max_error"]:.2e} K', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)

        # Adjust layout and save
        plt.tight_layout()

        # Create Figures directory if it doesn't exist
        figures_dir = os.path.join(os.path.dirname(__file__), 'Figures')
        os.makedirs(figures_dir, exist_ok=True)

        # Save plot
        plot_filename = os.path.join(figures_dir, f'{case_name}_comparison.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved plot: {plot_filename}")


    def create_base_input(self, params, boundary_config):
        '''Create base input dictionary with common parameters'''
        base_input = {
            'Materials': {
                'A': {
                    'k': 1.0,           # Thermal conductivity
                    'rho': 1000.0,       # Density
                    'cp': params['c'],   # Specific heat
                }
            },
            'Domain Table': {
                'Material Name': ['A'],
                'Thickness': [0.02],    # Small domain for lumped approximation
                'dx': [0.02],         # Multiple control volumes
            },
            'Boundary': boundary_config,
            'Time': {
                'Run Time': params['time_span'][1],
                'T Initial': params['T_initial'],
                'Output Frequency': 10
            },
            'Other': {
                'Y Dimension': params['L_y'],
                'Z Dimension': params['L_z'],
            }
        }
        return base_input


    def create_convection_input(self, params):
        '''Create input dictionary for convection-only test case'''
        boundary_config = {
            'External': {
                'Type': 'convection',
                'h': params['h'],
                'T': params['T_inf']
            },
            'Left': {
                'Type': 'convection',
                'h': params['h'],
                'T': params['T_inf']
            },
            'Right': {'Type': 'adiabatic'}
        }
        return self.create_base_input(params, boundary_config)


    def create_flux_input(self, params):
        '''Create input dictionary for constant heat flux test case'''
        boundary_config = {
            'External': {'Type': 'adiabatic'},
            'Left': {
                'Type': 'heat flux',
                'Flux': params['q_flux']
            },
            'Right': {'Type': 'adiabatic'}
        }
        return self.create_base_input(params, boundary_config)


    def create_combined_bc_input(self, params):
        '''Create input dictionary for combined convection-heat flux test case'''
        boundary_config = {
            'External': {
                'Type': 'convection',
                'h': params['h'],
                'T': params['T_inf']
            },
            'Left': [
                {
                    'Type': 'convection',
                    'h': params['h'],
                    'T': params['T_inf']
                },
                {
                    'Type': 'heat flux',
                    'Flux': params['q_flux']
                }
            ],
            'Right': {'Type': 'adiabatic'}
        }
        return self.create_base_input(params, boundary_config)


    def create_base_params(self, L_x=0.02, L_y=0.01, L_z=0.01):
        '''Create base parameter dictionary with common values'''
        return {
            'A_ext': 2*(L_x*L_y + L_x*L_z),     # Total surface area for the external faces
            'A_cross': L_y * L_z,               # Cross sectional area
            'm': L_x*L_y*L_z*1000.0,            # kg (rho * V = 1000 * L_x*L_y*L_z)
            'c': 1000.0,                        # J/kg/K
            'time_span': [0, 10],
            'L_x': L_x,                         # X dimension
            'L_y': L_y,                         # Y dimension
            'L_z': L_z                          # Z dimension
        }


    def run_simulation_with_input(self, input_dict):
        '''Run simulation with given input dictionary and return results'''
        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(input_dict, f)
            temp_file = f.name

        try:
            # Run model
            model = main_fv.lim1tr_model(temp_file)
            eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()

            # Extract results
            t_sim = data_man.data_dict['Time'].flatten()
            T_sim = data_man.data_dict['Temperature'].flatten()

            # For lumped analysis, use average temperature across all spatial points
            # Reshape to (time_steps, spatial_points) and take average over spatial dimension
            n_time = len(t_sim)
            n_space = len(T_sim) // n_time
            T_avg = T_sim.reshape((n_time, n_space)).mean(axis=1)

            return {
                'time': t_sim,
                'temperature': T_avg,
                'model': model
            }
        finally:
            # Clean up temporary file
            os.unlink(temp_file)


    def test_convection_only(self):
        '''Test convection-only BC against analytical solution'''
        print('\nTesting convection-only BC with analytical solution...')

        # Setup parameters using base and add convection-specific values
        params = self.create_base_params()
        params.update({
            'T_initial': 350.0,  # K
            'T_inf': 293.0,      # K
            'h': 10.0,           # W/m^2/K
        })

        # Create input file with convection BC
        input_dict = self.create_convection_input(params)

        # Run simulation
        results = self.run_simulation_with_input(input_dict)

        # Compare with analytical solution
        t_values = results['time']
        T_numerical = results['temperature']
        T_analytical = self.convection_analytical(
            params['T_initial'], params['T_inf'], params['h'],
            params['A_ext'] + params['A_cross'], params['m'], params['c'], t_values
        )

        # Calculate error metrics
        error_metrics = self.calculate_errors(T_numerical, T_analytical)

        print(f"Convection test - RMSE: {error_metrics['rmse']:.2e}, Max Error: {error_metrics['max_error']:.2e}")

        # Plot results
        self.plot_results(t_values, T_numerical, T_analytical, 'convection_only', error_metrics)

        # Assert within tolerance
        self.assertTrue(error_metrics['rmse'] < 1e-1,
                       f'Failed with RMSE {error_metrics["rmse"]:.2e}')
        self.assertTrue(error_metrics['max_error'] < 1e-0,
                       f'Failed with Max Error {error_metrics["max_error"]:.2e}')


    def test_heat_flux_only(self):
        '''Test constant heat flux BC against analytical solution'''
        print('\nTesting constant heat flux BC with analytical solution...')

        # Setup parameters using base and add flux-specific values
        params = self.create_base_params()
        params.update({
            'T_initial': 300.0,     # K
            'q_flux': 4000.0,       # W/m^2
        })

        # Create input with flux BC on left boundary
        input_dict = self.create_flux_input(params)

        # Run simulation
        results = self.run_simulation_with_input(input_dict)

        # Compare with analytical solution
        t_values = results['time']
        T_numerical = results['temperature']
        T_analytical = self.flux_analytical(
            params['T_initial'], params['q_flux'], params['A_cross'],
            params['m'], params['c'], t_values
        )

        # Calculate error metrics
        error_metrics = self.calculate_errors(T_numerical, T_analytical)

        print(f"Flux test - RMSE: {error_metrics['rmse']:.2e}, Max Error: {error_metrics['max_error']:.2e}")

        # Plot results
        self.plot_results(t_values, T_numerical, T_analytical, 'flux_only', error_metrics)

        # Assert within tolerance
        self.assertTrue(error_metrics['rmse'] < 1e-1,
                       f'Failed with RMSE {error_metrics["rmse"]:.2e}')
        self.assertTrue(error_metrics['max_error'] < 5e-1,
                       f'Failed with Max Error {error_metrics["max_error"]:.2e}')


    def test_combined_convection_heat_flux(self):
        '''Test combined convection and heat flux BCs'''
        print('\nTesting combined convection and heat flux BCs with analytical solution...')

        # Setup parameters using base and add combined-specific values
        params = self.create_base_params()
        params.update({
            'T_initial': 500.0, # K
            'T_inf': 293.0,     # K
            'h': 5.0,           # W/m^2/K
            'q_flux': 10000.0,   # W/m^2
        })

        # Create input with multiple BCs
        input_dict = self.create_combined_bc_input(params)

        # Run simulation
        results = self.run_simulation_with_input(input_dict)

        # Compare with analytical solution
        t_values = results['time']
        T_numerical = results['temperature']
        T_analytical = self.combined_analytical(
            params['T_initial'], params['T_inf'], params['h'],
            params['A_ext'] + params['A_cross'], params['A_cross'], params['q_flux'],
            params['m'], params['c'], t_values
        )

        # Calculate error metrics
        error_metrics = self.calculate_errors(T_numerical, T_analytical)

        print(f"Combined test - RMSE: {error_metrics['rmse']:.2e}, Max Error: {error_metrics['max_error']:.2e}")

        # Plot results
        self.plot_results(t_values, T_numerical, T_analytical, 'combined_convection_flux', error_metrics)

        # Assert within tolerance (higher tolerance for combined approximation)
        self.assertTrue(error_metrics['rmse'] < 1e-11,
                       f'Failed with RMSE {error_metrics["rmse"]:.2e}')
        self.assertTrue(error_metrics['max_error'] < 1e-11,
                       f'Failed with Max Error {error_metrics["max_error"]:.2e}')
