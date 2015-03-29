#!/usr/bin/env python3

from setuptools import setup, find_packages


setup(
    name="sym2num",
    version="0.1.dev1",
    packages=find_packages(),
    install_requires=["attrdict", "numpy", "pystache", "sympy"],
    test_requires=["pytest"],
    
    # metadata for upload to PyPI
    author="Dimas Abreu Dutra",
    author_email="dimasadutra@gmail.com",
    description="Sympy to numpy code generator.",
    license="MIT",
    url="http://github.com/dimasad/sym2num",
)
