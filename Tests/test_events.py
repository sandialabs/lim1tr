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

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../Source'))

import main_fv
import boundary
import boundary_types
import events
from unit_mocks import grid_mock


class MockSys:
    def __init__(self, n_tot):
        self.RHS = np.zeros(n_tot)


class MockTracker:
    def __init__(self, flags):
        self.cell_in_tr = np.asarray(flags, dtype=bool)


class RecordAction:
    def __init__(self):
        self.calls = []


    def execute(self, t):
        self.calls.append(t)


class event_tests(unittest.TestCase):
    # Conditions
    def test_temperature_condition_single_node(self):
        cond = events.temperature_condition([0], 350.0)
        self.assertFalse(cond.check(0.0, np.array([349.0, 400.0]), None))
        self.assertTrue(cond.check(0.0, np.array([350.0, 400.0]), None))


    def test_temperature_condition_interface_mean(self):
        cond = events.temperature_condition([1, 2], 350.0)
        self.assertTrue(cond.check(0.0, np.array([0.0, 340.0, 362.0]), None))
        self.assertFalse(cond.check(0.0, np.array([0.0, 340.0, 358.0]), None))


    def test_tr_cell_condition(self):
        tracker = MockTracker([False, True])
        self.assertFalse(events.tr_cell_condition(0).check(0.0, None, tracker))
        self.assertTrue(events.tr_cell_condition(1).check(0.0, None, tracker))


    # Actions
    def test_deactivate_action(self):
        bc = boundary_types.end_bc(np.full(3, 0.1), 'Left')
        events.deactivate_bc_action(bc).execute(1.0)
        self.assertFalse(bc.active)


    def test_activate_action_resets_temporal_clock(self):
        flux_bc = boundary_types.end_flux(np.full(3, 0.1), 'Left')
        flux_bc.flux = 0.0
        flux_bc.setup_params()
        temporal = boundary_types.temporal_boundary(flux_bc)
        temporal.active = False
        events.activate_bc_action(temporal).execute(4.0)
        self.assertTrue(temporal.active)
        self.assertEqual(temporal.time_offset, 4.0)


    def test_activate_action_on_plain_bc(self):
        bc = boundary_types.end_bc(np.full(3, 0.1), 'Left')
        bc.active = False
        events.activate_bc_action(bc).execute(2.0)
        self.assertTrue(bc.active)


    def test_reset_time_through_nested_wrappers(self):
        '''reset_time on the outermost wrapper must reach the inner
        temporal_boundary so its params evaluate at t - time_offset.'''
        flux_bc = boundary_types.end_flux(np.full(3, 0.1), 'Left')
        flux_bc.flux = 0.0
        flux_bc.setup_params()
        temporal = boundary_types.temporal_boundary(flux_bc)
        temporal.add_param('flux', boundary_types.ramp_function(10.0, 100.0))
        outer = boundary_types.timed_boundary(temporal, 1.0e6)
        outer.reset_time(5.0)
        sys_mock = MockSys(3)
        outer.apply(sys_mock, None, 7.0, None)
        self.assertAlmostEqual(sys_mock.RHS[0], 100.0 + 2.0*10.0)


    # Rules
    def test_one_shot_latches(self):
        cond = events.temperature_condition([0], 350.0)
        trig, rel = RecordAction(), RecordAction()
        rule = events.event_rule(cond, [trig], [rel], True)
        rule.update(1.0, np.array([300.0]), None)
        rule.update(2.0, np.array([360.0]), None)
        rule.update(3.0, np.array([300.0]), None)
        rule.update(4.0, np.array([360.0]), None)
        self.assertEqual(trig.calls, [2.0])
        self.assertEqual(rel.calls, [])


    def test_toggle_trigger_and_release(self):
        cond = events.temperature_condition([0], 350.0)
        trig, rel = RecordAction(), RecordAction()
        rule = events.event_rule(cond, [trig], [rel], False)
        rule.update(1.0, np.array([300.0]), None)
        rule.update(2.0, np.array([360.0]), None)
        rule.update(3.0, np.array([360.0]), None)
        rule.update(4.0, np.array([300.0]), None)
        rule.update(5.0, np.array([360.0]), None)
        self.assertEqual(trig.calls, [2.0, 5.0])
        self.assertEqual(rel.calls, [4.0])


    # Boundary manager
    def test_bc_manager_skips_inactive(self):
        bc_man = boundary.bc_manager(grid_mock())
        flux_bc = boundary_types.end_flux(bc_man.dx_arr, 'Left')
        flux_bc.flux = 50.0
        flux_bc.setup_params()
        flux_bc.user_name = 'F'
        bc_man.register_bc(flux_bc)
        sys_mock = MockSys(bc_man.n_tot)
        bc_man.apply(sys_mock, None, 0.0, None)
        self.assertAlmostEqual(sys_mock.RHS[0], 50.0)
        bc_man.get_by_name('F').active = False
        bc_man.apply(sys_mock, None, 0.0, None)
        self.assertAlmostEqual(sys_mock.RHS[0], 50.0)


    # Parser
    def _make_model(self):
        file_name = os.path.join(os.path.dirname(__file__), 'Inputs', 'simple_cond.yaml')
        return main_fv.lim1tr_model(file_name)


    def test_parse_events_block(self):
        model = self._make_model()
        cap = model.parser.cap_dict
        cap['Boundary']['Left']['Name'] = 'LeftBC'
        cap['Boundary']['Right']['Name'] = 'RightBC'
        cap['Events'] = {'Warm Up': {
            'Condition': {'Type': 'Temperature', 'T Location': 0, 'T Cutoff': 400.0},
            'On Trigger': {'Activate BC': ['LeftBC', 'RightBC']}}}
        mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.parser.apply_parse()
        self.assertEqual(len(bc_man.event_man.rules), 1)
        rule = bc_man.event_man.rules[0]
        self.assertEqual(rule.condition.node_indices, [0])
        self.assertEqual(len(rule.trigger_actions), 2)
        # BCs targeted by Activate BC on trigger start inactive
        self.assertFalse(bc_man.get_by_name('LeftBC').active)
        self.assertFalse(bc_man.get_by_name('RightBC').active)


    def test_parse_right_end_location(self):
        model = self._make_model()
        cap = model.parser.cap_dict
        cap['Boundary']['Right']['Name'] = 'RightBC'
        cap['Events'] = {'Cool Down': {
            'Condition': {'Type': 'Temperature', 'T Location': 1, 'T Cutoff': 400.0},
            'On Trigger': {'Deactivate BC': 'RightBC'}}}
        mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.parser.apply_parse()
        rule = bc_man.event_man.rules[0]
        self.assertEqual(rule.condition.node_indices, [grid_man.n_tot - 1])
        self.assertTrue(bc_man.get_by_name('RightBC').active)


    def test_no_events_block(self):
        model = self._make_model()
        mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.parser.apply_parse()
        self.assertIsNone(bc_man.event_man)


    def test_bad_bc_name_raises(self):
        model = self._make_model()
        model.parser.cap_dict['Events'] = {'Bad Name': {
            'Condition': {'Type': 'Temperature', 'T Location': 0, 'T Cutoff': 400.0},
            'On Trigger': {'Deactivate BC': ['Nope']}}}
        with self.assertRaises(KeyError):
            model.parser.apply_parse()


    def test_cell_tr_without_reactions_raises(self):
        model = self._make_model()
        model.parser.cap_dict['Boundary']['Left']['Name'] = 'LeftBC'
        model.parser.cap_dict['Events'] = {'No Rxn': {
            'Condition': {'Type': 'Cell TR', 'Cell Index': 0},
            'On Trigger': {'Deactivate BC': ['LeftBC']}}}
        with self.assertRaises(ValueError):
            model.parser.apply_parse()


    def test_release_with_one_shot_raises(self):
        model = self._make_model()
        model.parser.cap_dict['Boundary']['Left']['Name'] = 'LeftBC'
        model.parser.cap_dict['Events'] = {'Latched Toggle': {
            'Condition': {'Type': 'Temperature', 'T Location': 0, 'T Cutoff': 400.0},
            'On Trigger': {'Deactivate BC': ['LeftBC']},
            'On Release': {'Activate BC': ['LeftBC']}}}
        with self.assertRaises(ValueError):
            model.parser.apply_parse()


    def test_unknown_condition_type_raises(self):
        model = self._make_model()
        model.parser.cap_dict['Events'] = {'Mystery': {
            'Condition': {'Type': 'Pressure', 'T Location': 0},
            'On Trigger': {}}}
        with self.assertRaises(ValueError):
            model.parser.apply_parse()


    def test_unknown_action_type_raises(self):
        model = self._make_model()
        model.parser.cap_dict['Boundary']['Left']['Name'] = 'LeftBC'
        model.parser.cap_dict['Events'] = {'Mystery': {
            'Condition': {'Type': 'Temperature', 'T Location': 0, 'T Cutoff': 400.0},
            'On Trigger': {'Explode BC': ['LeftBC']}}}
        with self.assertRaises(ValueError):
            model.parser.apply_parse()


    # Integration
    def test_heater_deactivation_integration(self):
        '''An adiabatic bar heated by a named flux BC until the left end
        reaches T Cutoff; the event shuts the heater off and the bar
        equilibrates instead of heating through the end time.'''
        print('\nTesting heater deactivation event on a transient solve...')
        file_name = os.path.join(os.path.dirname(__file__), 'Inputs', 'event_heater.yaml')
        model = main_fv.lim1tr_model(file_name)
        eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()
        T_sol = data_man.data_dict['Temperature']

        # Heater was deactivated by the rule
        self.assertFalse(bc_man.get_by_name('Heater').active)
        # Left end reached the cutoff with at most one step of overshoot
        self.assertGreaterEqual(np.max(T_sol[:, 0]), 350.0)
        self.assertLess(np.max(T_sol[:, 0]), 355.0)
        # Bar equilibrated well below continued-heating temperature (~448 K)
        self.assertLess(np.max(T_sol[-1, :]), 340.0)
        self.assertGreater(np.min(T_sol[-1, :]), 310.0)
        self.assertAlmostEqual(np.max(np.abs(T_sol[-1, :] - T_sol[-2, :])), 0.0, places=3)


    def test_heater_swap_integration(self):
        '''A heater at the left end raises the local temperature until it
        crosses T Cutoff. The same event deactivates the heater and, in
        the same trigger, activates a convection BC at the right end that
        starts inactive. The bar then relaxes back to the convection BC's
        ambient temperature instead of continuing to heat, exercising
        Activate BC and Deactivate BC together in one transient solve.'''
        print('\nTesting heater deactivation + cooler activation event on a transient solve...')
        file_name = os.path.join(os.path.dirname(__file__), 'Inputs', 'event_heater_swap.yaml')
        model = main_fv.lim1tr_model(file_name)
        eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()
        T_sol = data_man.data_dict['Temperature']

        # Heater was deactivated and Cooler was activated by the rule
        self.assertFalse(bc_man.get_by_name('Heater').active)
        self.assertTrue(bc_man.get_by_name('Cooler').active)

        # Left end reached the cutoff with at most one step of overshoot
        self.assertGreaterEqual(np.max(T_sol[:, 0]), 350.0)
        self.assertLess(np.max(T_sol[:, 0]), 352.0)

        # With the heater off and the cooler on, the bar relaxes back to
        # the cooler's ambient temperature and reaches steady state,
        # rather than continuing to heat through the end time
        self.assertAlmostEqual(T_sol[-1, 0], 298.15, places=1)
        self.assertAlmostEqual(T_sol[-1, -1], 298.15, places=1)
        self.assertAlmostEqual(np.max(np.abs(T_sol[-1, :] - T_sol[-2, :])), 0.0, places=3)


    def test_heater_swap_integration_adaptive_step(self):
        '''Same scenario as test_heater_swap_integration, but with Spitfire's
        adaptive step size controller instead of a fixed dt. Output Frequency
        is bumped up so the dense-output history is still fine enough to
        capture the cutoff crossing and the post-swap cooldown.'''
        print('\nTesting heater deactivation + cooler activation event with adaptive time stepping...')
        file_name = os.path.join(os.path.dirname(__file__), 'Inputs', 'event_heater_swap.yaml')
        model = main_fv.lim1tr_model(file_name)
        del model.parser.cap_dict['Time']['dt']
        model.parser.cap_dict['Time']['Output Frequency'] = 100
        eqn_sys, cond_man, mat_man, grid_man, bc_man, reac_man, data_man, time_opts = model.run_model()
        T_sol = data_man.data_dict['Temperature']

        # Heater was deactivated and Cooler was activated by the rule
        self.assertFalse(bc_man.get_by_name('Heater').active)
        self.assertTrue(bc_man.get_by_name('Cooler').active)

        # Left end reached the cutoff. Dense-output interpolation across
        # the BC-switch kink can slightly undershoot the true accepted-step
        # peak, so the bound is looser than the fixed-step version.
        self.assertGreater(np.max(T_sol[:, 0]), 349.0)
        self.assertLess(np.max(T_sol[:, 0]), 351.0)

        # With the heater off and the cooler on, the bar relaxes back to
        # the cooler's ambient temperature and reaches steady state,
        # rather than continuing to heat through the end time
        self.assertAlmostEqual(T_sol[-1, 0], 298.15, places=1)
        self.assertAlmostEqual(T_sol[-1, -1], 298.15, places=1)
        self.assertAlmostEqual(np.max(np.abs(T_sol[-1, :] - T_sol[-2, :])), 0.0, places=3)


if __name__ == '__main__':
    unittest.main()
