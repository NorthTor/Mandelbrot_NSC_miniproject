"""
Author: Tor Kaufmann Gjerde
May 2021

See file: "cython_naive_mandelbrot.pyx" for clarification
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('cython_naive_mandelbrot.pyx'))
