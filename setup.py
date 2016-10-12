#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='sklearn-deap',
    version='0.1.5',
    author='Rodrigo',
    author_email='',
    description='Use evolutionary algorithms instead of gridsearch in scikit-learn.',
    url='https://github.com/rsteca/sklearn-deap',
    download_url='https://github.com/rsteca/sklearn-deap/tarball/0.1.5',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
    ],
    package_dir={'': '.'},
    packages=find_packages('.'),
    install_requires=[
        'numpy>=1.9.3',
        'scipy>=0.16.0',
        'deap>=1.0.2',
        'scikit-learn>=0.16.1',
    ],
)
