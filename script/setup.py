# -*- coding: utf-8 -*-
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "reactIDR.cython.fast_param_fit",  # モジュール名（パッケージパス付き）
        sources=["reactIDR/cython/fast_param_fit.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="reactIDR",
    version="2.0.0",
    url='https://github.com/carushi/reactIDR/',
    author='carushi',
    author_email='1845273+carushi@users.noreply.github.com',
    license="GPL2",
    ext_modules=cythonize(extensions, language_level=3),
    zip_safe=False,
)
