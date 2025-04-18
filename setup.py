# -*- coding: utf-8 -*-
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "reactIDR.cython.fast_param_fit",
        sources=["reactIDR/cython/fast_param_fit.pyx"],
        include_dirs=[numpy.get_include()],
    )
]


setup(
    name="reactIDR",
    version="2.0.1",
    description="Implementation of the IDR computation for RNA reactivity data.",
    author="carushi",
    author_email="1845273+carushi@users.noreply.github.com",
    license="GPL2",
    packages=["reactIDR", "reactIDR.cython"],
    ext_modules=ext_modules,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=2.0.2",
        "scipy>=1.13.1",
        "pandas>=2.2.3",
        "matplotlib"
    ],
    entry_points={
        "console_scripts": [
            "reactIDR = reactIDR.main:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)"
    ],
)


