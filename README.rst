Metamodeling Framework
================================

|build| |docs|


The metamodeling framework was created to build models to use for estimating commercial building energy loads. The framework currently supports linear models, random forests, and support vector regressions. The framework handles the building, evalulation, and validation of the models. During each set of the process, the framework exports diagnostic data for the user to evaluate the performance of the reduced order models. In addition to building, evaluating, and validating the reduced order models, the framework is able to load previously persisted model to be used in third-party applications (e.g. Modelica).

The project was initially developed focusing on evaluating ambient loop district heating and cooling systems. As a result, there are several hard coded methods designed to evaluate and validate building energy modeling data. These are planned to be removed and made more generic in the coming months.

This documentation will discuss how to inspect, build, evaluate, and validate a simple dataset focused on commercial building energy consumption. The documentation will also demonstrate how to load and run an already built metamodel to be used to approximate building energy loads.

------------
Instructions
------------

The framework requires `Python 3 <https://www.python.org/>`_. After installing Python and configuring Python 3, the framework can be installed from source code (recommended) or from `PyPI <https://pypi.python.org/pypi>`_.

Installation from Source
========================

1) Install Python and pip

2) Clone this repository

3) Install the Python dependencies

    .. code-block:: sql
        :linenos:

        pip install -r requirements.txt

4) (Optional) install graphviz to visualize decision trees

    * OSX: :code:`brew install graphviz`


Building Example Models
=======================

A small office example has been included with the source code under the tests directory. The small office includes 3,300 hourly samples of building energy consumption with several characteristics for each sample. The example shown here is only the basics, for further instructions view the complete documentation on `readthedocs <https://metamodeling-framework.readthedocs.io/en/develop/>`_.

    .. code-block:: bash
        :linenos:

        ./meta-runner build -f metamodeling/tests/smoff_test/metamodels.json -a smoff_test
        ./meta-runner evaluate -f metamodeling/tests/smoff_test/metamodels.json -a smoff_test
        ./meta-runner validate -f metamodeling/tests/smoff_test/metamodels.json -a smoff_test

Installation from PyPI
======================

Not yet complete.

Example Repository
==================

An example repository was developed using the Metamodeling Framework to evaluate the results of OpenStudio using PAT. There are several repositories to generate the datasets; however, the first link below contains a basic dataset in order to demonstrate the functionality of the Metamodeling Framework.

* `Ambient Loop Metamodels <https://github.com/nllong/Ambient-Loop-Metamodels>`_

The two repositories below were used to generate the OpenStudio/EnergyPlus models used for the Metamodeling FRamework.

* `OpenStudio's Parametric Analysis Tool Projects <https://github.com/nllong/ambient-loop-pat-projects>`_
* `OpenStudio Measures <https://github.com/nllong/ambient-loop-measures>`_

To Dos
======

* Configure better CLI
* Allow for CLI to save results in specific location
* Remove downloaded simulation data from repository
* Write test for running the analysis_definition (currently untested!)

.. |build| image:: https://travis-ci.org/nllong/metamodeling-framework.svg?branch=develop
    :target: https://travis-ci.org/nllong/metamodeling-framework

.. |docs| image:: https://readthedocs.org/projects/reduced-order-modeling-framework/badge/?version=latest
    :target: https://metamodeling-framework.readthedocs.io/en/develop/?badge=develop
    :alt: Documentation Status
