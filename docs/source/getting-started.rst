Getting Started
***************

.. toctree::

The Metamodeling Framework is designed to help users build, evaluate, validate, and run reduced order models.
The image below shows the typical workflow and the required data. Each of the blue boxes
represent a process and the green boxes represent either an input dataset or a output data.

In order to run the build method, the user must supply the data in CSV format with an accompanying
JSON file which describes a) the build options, b) the response variables, and c) the covariates.
An explanation and example of how the metadata JSON config file looks is shown in
:doc:`example metadata json file <metadata_json_ex>`.

.. image:: images/fig_build_flow_chart.png


The four main functions of the meta-runner.py file includes:

1) Inspect

    Load the results dataframe and create a resulting dataframe (and CSV) describing the data.
    This is useful when determining what is in the dataframe and what the covariates and responses
    should be. This also calculates the means for all the variables which can be used to set
    as the default values for when running a parametric sweep with the resulting metamodel that
    is generated.

    Each model that is generated will create the statistics summary in the data directory.

     .. code-block:: bash

          -f FILE, --file FILE  Metadata file to use
          -a ANALYSIS_MONIKER, --analysis-moniker ANALYSIS_MONIKER
                                Name of the Analysis Model
          -m [{LinearModel,RandomForest,SVR}], --model-type [{LinearModel,RandomForest,SVR}]
                                Type of model to build
          -d DOWNSAMPLE, --downsample DOWNSAMPLE
                                Specific down sample value

    .. code-block:: bash

        ./meta-runner inspect -a smoff_test

2) Build

    Use the build positional argument to build a new reduced order model as defined in the
    metamodels.json file. There are several arguments that can be passed with the build command
    including:

    .. code-block:: bash

          -f FILE, --file FILE  Metadata file to use
          -a ANALYSIS_MONIKER, --analysis-moniker ANALYSIS_MONIKER
                                Name of the Analysis Model
          -m [{LinearModel,RandomForest,SVR}], --model-type [{LinearModel,RandomForest,SVR}]
                                Type of model to build
          -d DOWNSAMPLE, --downsample DOWNSAMPLE
                                Specific down sample value

    .. code-block:: bash

        ./meta-runner build -a smoff_test

3) Evaluate

    Use the build positional argument to build a new reduced order model as defined in the
    metamodels.json file. There are several arguments that can be passed with the build command
    including:

    .. code-block:: bash

          -f FILE, --file FILE  Metadata file to use
          -a ANALYSIS_MONIKER, --analysis-moniker ANALYSIS_MONIKER
                                Name of the Analysis Model
          -m [{LinearModel,RandomForest,SVR}], --model-type [{LinearModel,RandomForest,SVR}]
                                Type of model to build
          -d DOWNSAMPLE, --downsample DOWNSAMPLE
                                Specific down sample value

    .. code-block:: bash

        ./meta-runner evaluate -a smoff_test


4) Validate

    Use the build positional argument to build a new reduced order model as defined in the
    metamodels.json file. There are several arguments that can be passed with the build command
    including:

    .. code-block:: bash

          -f FILE, --file FILE  Metadata file to use
          -a ANALYSIS_MONIKER, --analysis-moniker ANALYSIS_MONIKER
                                Name of the Analysis Model
          -m [{LinearModel,RandomForest,SVR}], --model-type [{LinearModel,RandomForest,SVR}]
                                Type of model to build
          -d DOWNSAMPLE, --downsample DOWNSAMPLE
                                Specific down sample value

    .. code-block:: bash

        ./meta-runner validate -a smoff_test

5) Run

    Use the build positional argument to build a new reduced order model as defined in the
    metamodels.json file. There are several arguments that can be passed with the build command
    including:

    .. code-block:: bash

        -ad ANALYSIS_DEFINITION, --analysis-definition ANALYSIS_DEFINITION
                                Definition of an analysis to run using the Metamodels
        -w WEATHER, --weather WEATHER
                                Weather file to run analysis-definition
        -o OUTPUT, --output OUTPUT
                                File to save the results to


    .. code-block:: bash

        ./meta-runner.py run -a smoff_parametric_sweep -m RandomForest -ad examples/smoff-one-year.json -w examples/lib/USA_CO_Golden-NREL.724666_TMY3.epw -d 0.15 -o output.csv
