Analysis Definition
===================

.. toctree::

   analysis_definition_ex_static
   analysis_definition_ex_epw
   analysis_definition_ex_single
   analysis_definition_ex_multiple
   analysis_definition_code


The analysis definition module is used for loading an already generated reduced order and
running a subsequent analysis. The input is a JSON file that defines each of the
covariates of interest. The analysis can take of

* Single value analysis, see :doc:`example <analysis_definition_ex_static>`

* Sweep values over a year (as defined by an EPW file), see :doc:`example <analysis_definition_ex_epw>`

* Sweep values over specified ranges for single variable, see :doc:`example <analysis_definition_ex_single>`

* Sweep values over specified ranges for multiple variable, see :doc:`example <analysis_definition_ex_multiple>`

To run an analysis with a JSON file, first load a metamodel, then load the analysis defintion.

.. code-block:: python

    from metamodeling.analysis_definition.analysis_definition import AnalysisDefinition
    from metamodeling.metamodels import Metamodels

