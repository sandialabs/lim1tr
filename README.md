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
    - "Interface Temperature": temperature at each material interface (2D array)
    - Species mass concentration indexed by user defined species names, if present (2D array)
- Dictionary of the rate of change of state variables with respect to time at each control volume and simulation time
    - "Time": simulation time (1D array)
    - "Reaction Temperature Rate": temperature rate due to chemical reactions (2D array)
    - "HRR": volumetric heat release rate from chemical reactions (2D array)
    - Species mass concentration rate indexed by user defined species names, if present (2D array)

### Requirements
- Python 2.7
- Numpy
- Scipy
- Pandas
- Matplotlib
- Numba
- PyYAML

### User Guide
The user guide for version 1.0 can be found at
https://www.sandia.gov/ess-ssl/wp-content/uploads/2021/10/LIM1TR_Guide_SAND2021-12281.pdf
