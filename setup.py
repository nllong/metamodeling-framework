# -*- coding: utf-8 -*-

import re

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

version = re.search(
    '^__version__\s*=\s*\'(.*)\'',
    open('rom/version.py').read(),
    re.M).group(1)

setup(
    # TODO: rename to Metamodeling Framework
    name='ROM Framework',
    version=version,
    description='Generate metamodels based on arbitrary CSV files',
    long_description=readme,
    author='Nicholas Long',
    author_email='nicholas.lee.long@gmail.com',
    url='https://github.com/nllong/ROM-Framework',
    license=license,
    python_requires='>=3',
    # If updating here, then make sure to update requirements.txt file as well.
    install_requires=[
        'scikit-learn==0.19.2',
        'matplotlib==2.2.3',
        'pandas==0.23.2',
        'seaborn==0.9.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages(exclude=('tests', 'docs')),
)
