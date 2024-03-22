from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os
os.chdir('./Source')

setup(
    ext_modules=cythonize(
        "tridiag_cython.pyx", compiler_directives={"language_level": "3"}
    ),
    include_dirs=[numpy.get_include()],
)