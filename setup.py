# -*- coding: utf-8 -*-
from setuptools import setup
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    name="reactIDR",
    version="1.1.0",
    url='https://github.com/carushi/reactIDR/',
    author='carushi',
    author_email='trumpet-lambda@hotmail.co.jp',
    license="GPL3",
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("fastpo", ["reactIDR/cython/fast_param_fit.pyx"], include_dirs=[numpy.get_include()], library_dirs=["reactIDR/"])],
    entry_points={
        "console_scripts": [
            "reactIDR = reactIDR.main"
            ]
        },
)
