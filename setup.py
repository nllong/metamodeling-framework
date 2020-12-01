# -*- coding: utf-8 -*-

import re

from setuptools import setup, find_packages

# from metamodeling import run

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

version = re.search(
    '^__version__\s*=\s*\'(.*)\'',
    open('metamodeling/version.py').read(),
    re.M).group(1)

setup(
    name='Metamodeling Framework',
    version=version,
    description='Generate metamodels based on arbitrary CSV files',
    long_description=readme,
    author='Nicholas Long',
    author_email='nicholas.lee.long@gmail.com',
    url='https://github.com/nllong/metamodeling-framework',
    license=license,
    python_requires='>=3',
    # If updating here, then make sure to update requirements.txt file as well.
    install_requires=[
        'matplotlib==3.2.1',
        'numpy==1.18.3',
        'pandas==1.0.3',
        'pyfiglet==0.8.post1',
        'requests==2.23.0',
        'scikit-learn==0.22.2',
        'scipy==1.4.1',
        'seaborn==0.11.',
        'seaborn-qqplot==0.5.0',
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
    scripts=['bin/metamodel.py'],
)

