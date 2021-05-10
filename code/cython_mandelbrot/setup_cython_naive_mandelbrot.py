"""
@author: Tor Kaufmann Gjerde  
May 201
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize('cython_naive_mandelbrot.pyx')) 
 