#!/usr/bin/env python3

from setuptools import setup, find_packages

DESCRIPTION = open("README.rst", encoding="utf-8").read()

CLASSIFIERS = '''\
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved
Operating System :: MacOS
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3 :: Only
Topic :: Scientific/Engineering
Topic :: Software Development'''

setup(
    name="sym2num",
    version="0.1.dev2",
    packages=find_packages(),
    install_requires=["attrdict", "jinja2", "numpy", "sympy"],
    tests_require=["pytest"],
    
    # metadata for upload to PyPI
    author="Dimas Abreu Archanjo Dutra",
    author_email="dimasad@ufmg.br",
    description="Sympy to numpy code generator.",
    long_description=DESCRIPTION,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    classifiers=CLASSIFIERS.split('\n'),
    license="MIT",
    url="http://github.com/cea-ufmg/sym2num",
)
