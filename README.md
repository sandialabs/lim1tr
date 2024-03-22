# LIM1TR: Lithium-Ion Modeling with 1-D Thermal Runaway

LIM1TR is a control volume code with 1D heat transport and reaction kinetics for modeling thermal runaway in Li-ion batteries.

### Setup
Install the following requirements, preferrably in a fresh environment. Begin with installing [Spitfire](https://github.com/sandialabs/Spitfire) following the [documentation](https://spitfire.readthedocs.io/en/latest/?badge=latest).

- Python >= 3.7
- [Spitfire](https://github.com/sandialabs/Spitfire)
    - compilers
    - setuptools
    - numpy
    - scipy
    - matplotlib
    - Cython
    - sphinx
    - numpydoc
    - gitpython
    - cantera
- pandas
- pyyaml

After installing the required packages, build the tridiagonal matrix algorithm from the main LIM1TR directory with:

`$ python setup.py build_ext --inplace`

### To Run  
`$ python main_fv.py input_file_name.yaml`

Setting up an alias for `python main_fv.py` is recommended.

### Output
Output is saved in the current working directory as a pickle file containing a list of the following objects:
- Dictionary of the .yaml input file
- Dictionary of the state variables at each control volume and simulation time
    - "Time": simulation time (1D array)
    - "Grid": location of the center of each control volume (1D array)
    - "Layer Map": a map of the first and last control volume indies in each layer (dictionary)
    - "Temperature": temperature (2D array)
    - "Interface Temperature": temperature at each material interface (2D array)
    - Species mass concentration indexed by user defined species names, if present (2D array)
- Dictionary of the rate of change of state variables with respect to time at each control volume and simulation time
    - "Time": simulation time (1D array)
    - "HRR": volumetric heat release rate from chemical reactions (2D array)
    - Species mass concentration rate indexed by user defined species names, if present (2D array)

### User Guide
The user guide for version 1.0 can be found at
https://www.sandia.gov/ess-ssl/wp-content/uploads/2021/10/LIM1TR_Guide_SAND2021-12281.pdf
