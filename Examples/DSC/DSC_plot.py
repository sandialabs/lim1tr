from __future__ import division
import numpy as np
import pandas as pd
import sys, os
import pickle as p
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['legend.fontsize'] = 24
mpl.rcParams['figure.figsize'] = (12., 10.)
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['lines.linewidth'] = 3


# Load input dictionary and output data
with open('2_rxn_output.p', 'r') as f:
    my_input, my_data, my_rate = p.load(f)

# Plot temperature vs time
plt.figure()
plt.plot(my_data['Time'], my_data['Temperature'] - 273.15)
plt.xlabel('Time (s)')
plt.ylabel(r'Temperature ($\mathrm{^o}$C)')
plt.savefig('Temperature.png', bbox_inches='tight')
plt.close()

# DSC heat release rate vs temperature
Y_SEI = my_input['Species']['Initial Mass Fraction'][2]
Y_C6Li = my_input['Species']['Initial Mass Fraction'][1]
rho = my_input['Materials']['A']['rho']
anode_density = rho*(Y_SEI + Y_C6Li)*1000

plt.figure()
plt.plot(my_data['Temperature'] - 273.15, my_rate['HRR']/anode_density)
plt.xlabel(r'Temperature ($\mathrm{^o}$C)')
plt.ylabel(r'Heat Flow (W/g Anode)')
plt.savefig('HRR.png', bbox_inches='tight')
plt.close()

# Species densities 
sei_keys = ['SEI', 'Salt1', 'AllGas']
anode_keys = ['C6Li', 'EC', 'Li2CO3', 'C6']

plt.figure()
for key in sei_keys:
    plt.plot(my_data['Time'], my_data[key], label=key)
for key in anode_keys:
    plt.plot(my_data['Time'], my_data[key], label=key)
plt.xlabel('Time (s)')
plt.ylabel('Density (kg/m3)')
plt.legend(loc=2)
plt.savefig('Densities.png', bbox_inches='tight')
plt.close()
