#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='evolutionary_search',
    version='0.1',
    author='Rodrigo',
    author_email='',
    description='',
    url='',
    download_url='',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 2 :: Only',
        'Programming Language :: Python',
    ],
    package_dir={'': '.'},
    packages=find_packages('.'),
    install_requires=[
        'numpy>=1.9.3',
        'scipy>=0.16.0',
        'bitstring>=3.1.3',
        'deap>=1.0.2',
        'scikit-learn>=0.16.1',
    ],
)
