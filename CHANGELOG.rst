Version 0.1.2 (Current Develop)
===============================

* Rename project to metamodeling framework
* Allow results_file to magically try and find the results in the post_process/{metamodel['name']}/simulation_results.csv}
* Move downloading of simulations to Python. In custom script call from metamodeling.post_process import OpenStudioServerAPI
* Remove Ruby and add post processing python code. User will need to write their own post processing scripts.
* Remove the ANOVA plots in LinearModel due to missing statsmodel package.
* Remove the "results_directory" key in the metamodel definition file. This was no longer used.
* Add random to the validation_id option in the metamodel definition
* Remove some hard coded validation checks such as calculating the total HVAC energy
* Do not remove inspection results from the data folder when running build.
* Rename ETSModel to Metamodel. Metamodels point to a Metamodel (not ETSModel)
* Update SciKit Learn to 0.22.2, Update NumPy to 1.18.3, and other dependencies.
* Use pipelines for Random Forest and enable categorical variables
* Fix EPW Day of week calculation. Add Day of Week (%A) and Day of Week Int (%w).
* Add seaborn-qqplot to requirements
* Fix bug passing downsampling value to validation (users needs to ensure there is a downsample key in the metamodel.json file)
* Remove the renaming of ETS and district energy variables. These now must be renamed in the postprocessing of the simulation results.

Version 0.1.1
=============

* Adds an "inspect" method to the rom-runner (rom.py) CLI
* Remove example files
* Updates the dependency (setup) file to ensure correct installation of dependencies
* Use Python 3's super syntax
* Move example runs to a unittest
* Add Python coverage
* Move Ruby post processing script to 'post_process' folder
* Add results_file to the metamodel.json config file to point to the CSV that needs to be loaded
* Moved unit tests to root directory
* Moved post-processing script (which is in Ruby) from the data directory to the post_process directory

Version 0.1.0
=============

* Initial Release
