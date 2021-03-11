# LIM1TR: Lithium-Ion Modeling with 1-D Thermal Runaway

LIM1TR is a control volume code with 1D heat transport and reaction kinetics for modeling thermal runaway in Li-ion batteries.

### To Run  
`$ python main_fv.py input_file_name.yaml`

### Output
Output is saved in a pickle file as a list of the following objects:
- Dictionary of the .yaml input file
- Dictionary of the state variables at each control volume and simulation time
    - "Time": simulation time (1D array)
    - "Grid": location of the center of each control volume (1D array)
    - "Temperature": temperature (2D array)
    - "Interface Temperature": temparature at each material interface (2D array)
    - Species mass concentration indexed by user defined species names, if present (2D array)
- Dictionary of the rate of change of state variables with respect to time at each control volume and simulation time
    - "Time": simulation time (1D array)
    - "Temperature Rate": temperature rate (2D array)
    - "HRR": volumetric heat release rate from chemical reactions (2D array)
    - Species mass concentration rate indexed by user defined species names, if present (2D array)

### Requirements
- Python 2.7
- Numpy
- Scipy
- Pandas
- Matplotlib
- Numba

A full user guide is in preparation.
