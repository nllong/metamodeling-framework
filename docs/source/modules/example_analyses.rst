Using the ROMs
==============

CLI Example
-----------

This example CLI file shows how a simple application can be built that reads in all the values
from the command line and reports the values of the responses passed.

.. code-block:: bash

    # Single response - with only setting the inlet temperature
    python analysis_cli_ex1.py -f smoff/metamodels.json -i 18

    # Multiple responses - with only setting the inlet temperature
    python analysis_cli_ex1.py -f smoff/metamodels.json -i 18 -r HeatingElectricity DistrictHeatingHotWaterEnergy

.. literalinclude:: ../../../examples/analysis_cli_ex1.py
   :linenos:

Analysis Example
----------------

Example analysis script demonstrating how to programatically load and run already persisted
reduced order models. This example loads two response variables (models) from the small office
random forest reduced order models. The loaded models are then passed through the
swee-temp-test.json analysis definition file. The analysis definition has few fixed
covariates and a few covariates with multiple values to run.

.. code-block:: bash

    python analysis_ex1.py

.. literalinclude:: ../../../examples/analysis_ex1.py
   :linenos:

Sweep Example
-------------

Example analysis script demonstrating how to programatically load and run already persisted
reduced order models using a weather file. This example is very similar to the analysis_ex1.py
excpet for the analysis.load_weather_file method. This method and the smoff-one-year.json file
together specify how to parse the weather file.

The second part of this script using seaborn to generate heatmaps of the two responses of
interest. The plots are stored in a child directory. Run the example by calling the following:

.. code-block:: bash

    python analysis_sweep_ex1.py

.. literalinclude:: ../../../examples/analysis_sweep_ex1.py
   :linenos:

Modelica Example
----------------

This example file shows how to load the models using a method based approach for use in Modelica.
The run_model takes only a list of numbers (int and floats). The categorical variables are
converted as needed in order to correctly populate the list of covariates in the dataframe.

For use in Modelica ake sure that the python path is set, such as by running
export PYTHONPATH=`pwd`

Call the following bash command shown below to run this example. This example runs as an
entrypoint; however, when connected to modelica the def run_model will be called directory. Also,
note that the run_model method loads the models every time it is called. This is non-ideal when
using this code in a timestep by timestep simulation. Work needs to be done to determine how to
load the reduced order models only once and call the reduced order model yhat methods each timestep.

.. code-block:: bash

        python analysis_modelica_ex1.py

.. literalinclude:: ../../../examples/analysis_modelica_ex1.py
   :linenos:
