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


class thermal_runaway_tracker:
    def __init__(self, n_tot, reaction_cells, t_rate_threshold=0.0):
        self.n_tot = n_tot
        self.reaction_cells = reaction_cells
        self.n_cells = len(reaction_cells)
        self.t_rate_threshold = t_rate_threshold

        self.node_in_tr = np.zeros(n_tot, dtype=bool)
        self.node_onset = np.full(n_tot, -1.0)
        self.node_end   = np.full(n_tot, -1.0)

        self.cell_in_tr = np.zeros(self.n_cells, dtype=bool)
        self.cell_onset = np.full(self.n_cells, -1.0)
        self.cell_end   = np.full(self.n_cells, -1.0)


    def update(self, t, T_rate_rxn):
        new_in_tr  = T_rate_rxn > self.t_rate_threshold
        onset_mask = new_in_tr & ~self.node_in_tr
        end_mask   = ~new_in_tr & self.node_in_tr
        self.node_onset[onset_mask & (self.node_onset < 0)] = t
        self.node_end[end_mask     & (self.node_end   < 0)] = t
        self.node_in_tr[:] = new_in_tr

        for j, cell in enumerate(self.reaction_cells):
            b1, b2 = cell.bounds
            new_cell_in_tr = bool(np.any(new_in_tr[b1:b2]))
            if new_cell_in_tr and not self.cell_in_tr[j] and self.cell_onset[j] < 0:
                self.cell_onset[j] = t
            if not new_cell_in_tr and self.cell_in_tr[j] and self.cell_end[j] < 0:
                self.cell_end[j] = t
            self.cell_in_tr[j] = new_cell_in_tr


    def get_output(self):
        return {
            'TR Node Onset': self.node_onset,
            'TR Node End':   self.node_end,
            'TR Cell Names': ['Cell {}'.format(j) for j in range(self.n_cells)],
            'TR Cell Onset': self.cell_onset,
            'TR Cell End':   self.cell_end,
        }
