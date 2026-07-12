########################################################################################
#                                                                                      #
#  Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).  #
#  Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains    #
#  certain rights in this software.                                                    #
#                                                                                      #
#  This software is released under the license detailed in the file, LICENSE.          #
#                                                                                      #
########################################################################################

import numpy as np


class tr_cell_condition:
    def __init__(self, cell_index):
        self.cell_index = cell_index


    def check(self, t, state, tr_tracker):
        return bool(tr_tracker.cell_in_tr[self.cell_index])


class temperature_condition:
    def __init__(self, node_indices, t_cutoff):
        self.node_indices = node_indices
        self.t_cutoff = t_cutoff


    def check(self, t, state, tr_tracker):
        return np.mean(state[self.node_indices]) >= self.t_cutoff


class deactivate_bc_action:
    def __init__(self, bc):
        self.bc = bc


    def execute(self, t):
        self.bc.active = False


class activate_bc_action:
    def __init__(self, bc):
        self.bc = bc


    def execute(self, t):
        self.bc.active = True
        self.bc.reset_time(t)


class event_rule:
    def __init__(self, condition, trigger_actions, release_actions, one_shot):
        self.condition = condition
        self.trigger_actions = trigger_actions
        self.release_actions = release_actions
        self.one_shot = one_shot
        self.prev_state = False
        self.done = False


    def update(self, t, state, tr_tracker):
        if self.done:
            return
        new_state = self.condition.check(t, state, tr_tracker)
        if new_state and not self.prev_state:
            for action in self.trigger_actions:
                action.execute(t)
            self.done = self.one_shot
        elif self.prev_state and not new_state:
            for action in self.release_actions:
                action.execute(t)
        self.prev_state = new_state


class event_manager:
    def __init__(self, rules):
        self.rules = rules


    def update(self, t, state, tr_tracker):
        for rule in self.rules:
            rule.update(t, state, tr_tracker)
