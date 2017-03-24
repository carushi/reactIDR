# -*- coding: utf-8 -*-
from setuptools import setup
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    name="reactIDR",
    version="0.1.0",
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("fastpo", ["fast_param_fit.pyx"], include_dirs=[numpy.get_include()])],
)