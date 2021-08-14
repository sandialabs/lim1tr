from __future__ import division
import numpy as np
import sys, os
import pickle as p
import matplotlib.pyplot as plt
plt.style.use('pres')


# Figures folder
fig_fold = './Figures/'
if not os.path.exists(fig_fold):
    os.mkdir(fig_fold)

# Load input dictionary and output data
with open('./3_rxn_output.p', 'rb') as f:
    my_cap, my_data, my_rate = p.load(f)

small_lab = ['C1-O', 'C1-C2', 'C2-C3', 'C3-C4', 'C4-C5', 'C5-O']

plt.figure()
for i in range(my_data['Interface Temperature'].shape[1]):
    p = plt.plot(my_data['Time'], my_data['Interface Temperature'][:,i]-273.15, label=small_lab[i])
plt.xlabel('Time (s)')
plt.ylabel(r'Temperature ($^\mathrm{o}$C)')
plt.legend()
plt.savefig(fig_fold + 'Temperature.png', bbox_inches='tight')
plt.close()
