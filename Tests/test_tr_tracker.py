########################################################################################
#                                                                                      #
#  Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).  #
#  Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains    #
#  certain rights in this software.                                                    #
#                                                                                      #
#  This software is released under the license detailed in the file, License.          #
#                                                                                      #
########################################################################################

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../Source'))

from tr_tracker import thermal_runaway_tracker


class MockCell:
    def __init__(self, b1, b2):
        self.bounds = (b1, b2)


class tr_tracker_tests(unittest.TestCase):
    def _make_tracker(self, n_tot, cells, threshold=100.0):
        return thermal_runaway_tracker(n_tot, cells, threshold)


    # Initialization
    def test_initial_state(self):
        '''All onset/end arrays start at -1 and no node is in TR.'''
        tracker = self._make_tracker(5, [MockCell(0, 3), MockCell(3, 5)])
        self.assertTrue(np.all(tracker.node_onset == -1.0))
        self.assertTrue(np.all(tracker.node_end == -1.0))
        self.assertTrue(np.all(~tracker.node_in_tr))
        self.assertTrue(np.all(tracker.cell_onset == -1.0))
        self.assertTrue(np.all(tracker.cell_end == -1.0))
        self.assertTrue(np.all(~tracker.cell_in_tr))


    def test_get_output_keys(self):
        '''get_output returns all five expected keys.'''
        tracker = self._make_tracker(4, [MockCell(0, 4)])
        out = tracker.get_output()
        for key in ('TR Node Onset', 'TR Node End', 'TR Cell Names',
                    'TR Cell Onset', 'TR Cell End'):
            self.assertIn(key, out)


    def test_cell_names(self):
        '''Cell names are index-based strings.'''
        tracker = self._make_tracker(6, [MockCell(0, 3), MockCell(3, 6)])
        names = tracker.get_output()['TR Cell Names']
        self.assertEqual(names, ['Cell 0', 'Cell 1'])


    # Node tracking — onset
    def test_no_tr_onset_stays_negative(self):
        '''Nodes below threshold never record an onset.'''
        tracker = self._make_tracker(3, [MockCell(0, 3)], threshold=100.0)
        rates = np.array([50.0, 50.0, 50.0])
        tracker.update(1.0, rates)
        tracker.update(2.0, rates)
        self.assertTrue(np.all(tracker.node_onset == -1.0))


    def test_node_onset_recorded_on_first_crossing(self):
        '''Onset time is set when a node first exceeds the threshold.'''
        tracker = self._make_tracker(3, [MockCell(0, 3)], threshold=100.0)
        below = np.array([50.0, 50.0, 50.0])
        above = np.array([200.0, 200.0, 200.0])
        tracker.update(1.0, below)
        tracker.update(2.0, above)
        self.assertTrue(np.all(tracker.node_onset == 2.0))


    def test_partial_onset(self):
        '''Only the node crossing the threshold gets its onset time set.'''
        tracker = self._make_tracker(3, [MockCell(0, 3)], threshold=100.0)
        tracker.update(5.0, np.array([50.0, 200.0, 50.0]))
        self.assertEqual(tracker.node_onset[0], -1.0)
        self.assertEqual(tracker.node_onset[1], 5.0)
        self.assertEqual(tracker.node_onset[2], -1.0)


    # Node tracking — end
    def test_node_end_recorded_on_recovery(self):
        '''End time is set when a node drops back below the threshold.'''
        tracker = self._make_tracker(2, [MockCell(0, 2)], threshold=100.0)
        tracker.update(1.0, np.array([200.0, 200.0]))
        tracker.update(3.0, np.array([50.0, 50.0]))
        self.assertTrue(np.all(tracker.node_end == 3.0))


    def test_no_end_if_still_in_tr(self):
        '''Nodes still in TR at simulation end have end == -1.'''
        tracker = self._make_tracker(2, [MockCell(0, 2)], threshold=100.0)
        tracker.update(1.0, np.array([200.0, 200.0]))
        out = tracker.get_output()
        self.assertTrue(np.all(out['TR Node End'] == -1.0))


    # First-event-only: onset and end are never overwritten
    def test_onset_not_overwritten_on_second_entry(self):
        '''A second TR entry does not overwrite the first onset time.'''
        tracker = self._make_tracker(1, [MockCell(0, 1)], threshold=100.0)
        tracker.update(1.0, np.array([200.0]))   # onset at t=1
        tracker.update(2.0, np.array([50.0]))    # recovery
        tracker.update(3.0, np.array([200.0]))   # re-enters TR
        self.assertEqual(tracker.node_onset[0], 1.0)


    def test_end_not_overwritten_on_second_recovery(self):
        '''A second recovery does not overwrite the first end time.'''
        tracker = self._make_tracker(1, [MockCell(0, 1)], threshold=100.0)
        tracker.update(1.0, np.array([200.0]))   # onset
        tracker.update(2.0, np.array([50.0]))    # end at t=2
        tracker.update(3.0, np.array([200.0]))   # re-enters
        tracker.update(4.0, np.array([50.0]))    # second recovery — should not overwrite
        self.assertEqual(tracker.node_end[0], 2.0)


    # Cell tracking
    def test_cell_onset_when_any_node_enters_tr(self):
        '''Cell onset fires when the first node in its bounds enters TR.'''
        tracker = self._make_tracker(4, [MockCell(0, 4)], threshold=100.0)
        rates = np.array([50.0, 50.0, 200.0, 50.0])
        tracker.update(7.0, rates)
        self.assertEqual(tracker.cell_onset[0], 7.0)


    def test_cell_stays_in_tr_while_any_node_is_in_tr(self):
        '''Cell stays in TR as long as at least one node in its bounds is in TR.'''
        tracker = self._make_tracker(3, [MockCell(0, 3)], threshold=100.0)
        tracker.update(1.0, np.array([200.0, 50.0, 50.0]))   # node 0 in TR
        tracker.update(2.0, np.array([50.0, 200.0, 50.0]))   # node 1 in TR, node 0 exits
        self.assertTrue(tracker.cell_in_tr[0])
        self.assertEqual(tracker.cell_end[0], -1.0)


    def test_cell_end_when_all_nodes_exit_tr(self):
        '''Cell end time is set when all nodes in its bounds exit TR.'''
        tracker = self._make_tracker(3, [MockCell(0, 3)], threshold=100.0)
        tracker.update(1.0, np.array([200.0, 200.0, 200.0]))
        tracker.update(5.0, np.array([50.0, 50.0, 50.0]))
        self.assertEqual(tracker.cell_end[0], 5.0)


    def test_two_cells_independent(self):
        '''Two non-overlapping cells are tracked independently.'''
        cells = [MockCell(0, 2), MockCell(2, 4)]
        tracker = self._make_tracker(4, cells, threshold=100.0)
        # Only left cell enters TR
        tracker.update(2.0, np.array([200.0, 200.0, 50.0, 50.0]))
        self.assertEqual(tracker.cell_onset[0], 2.0)
        self.assertEqual(tracker.cell_onset[1], -1.0)
        # Only right cell enters TR later
        tracker.update(4.0, np.array([50.0, 50.0, 200.0, 200.0]))
        self.assertEqual(tracker.cell_onset[1], 4.0)
        # Left cell has ended
        self.assertEqual(tracker.cell_end[0], 4.0)


    def test_cell_onset_not_overwritten(self):
        '''Cell onset is never overwritten after first entry.'''
        tracker = self._make_tracker(2, [MockCell(0, 2)], threshold=100.0)
        tracker.update(1.0, np.array([200.0, 200.0]))
        tracker.update(2.0, np.array([50.0, 50.0]))
        tracker.update(3.0, np.array([200.0, 200.0]))
        self.assertEqual(tracker.cell_onset[0], 1.0)


    def test_cell_no_onset_when_no_nodes_in_tr(self):
        '''Cell onset remains -1 if no nodes in bounds ever exceed threshold.'''
        cells = [MockCell(0, 2), MockCell(2, 4)]
        tracker = self._make_tracker(4, cells, threshold=100.0)
        # Only right-cell nodes exceed threshold
        tracker.update(1.0, np.array([50.0, 50.0, 200.0, 200.0]))
        self.assertEqual(tracker.cell_onset[0], -1.0)
        self.assertEqual(tracker.cell_onset[1], 1.0)


if __name__ == '__main__':
    unittest.main()
