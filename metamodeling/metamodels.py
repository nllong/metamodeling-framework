# -*- coding: utf-8 -*-
"""
.. moduleauthor:: Nicholas Long (nicholas.l.long@colorado.edu, nicholas.lee.long@gmail.com)
"""
import gc
import json
import os
import re
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .shared import unpickle_file, apply_cyclic_transform


class DuplicateColumnName(Exception):
    pass


class ETSModel:
    def __init__(self, response_name, model_file, scaler_file=None):
        """
        Load the model from a pandas pickled dataframe.

        :param response_name: str, The response to load (e.g. ETSOutletTemperature).
        :param model_file: str, The pickled model file path.
        :param scaler_file: str, The scaler file path.
        """

        self.response_name = response_name
        self.model_file = model_file
        self.scaler_file = scaler_file
        if os.path.exists(model_file) and os.path.isfile(model_file):
            gc.disable()
            self.model = unpickle_file(model_file)
            gc.enable()
        else:
            raise Exception("File not found, unable to load: %s" % model_file)

        if os.path.exists(scaler_file) and os.path.isfile(scaler_file):
            gc.disable()
            self.scalers = unpickle_file(scaler_file)
            gc.enable()
        else:
            self.scalers = None

    def yhat(self, data):
        """
        Run predict on supplied data.

        :param data: array, Values to predict on. The format is dependent on the model.
        e.g. [[month, hour, dayofweek, t_outdoor, rh, inlet_temp]]
        :return: array, Model predictions.
        """

        # Transform the feature data if scaler file exists
        if self.scalers:
            data[data.columns] = self.scalers['features'].transform(data[data.columns])

        predictions = self.model.predict(data)

        # Inverse transform out the response data
        if self.scalers:
            predictions = self.scalers[self.response_name].inverse_transform(predictions)

        return predictions

    def __str__(self):
        """
        Create string representation of model file.
        """
        return self.model_file


class Metamodels(object):
    def __init__(self, filename):
        self.filename = None
        self.file = None
        self.set_i = None
        self.load_file(filename)
        self.models = {}
        self.rom_type = None

    def load_file(self, filename):
        """
        Parse the file that defines the ROMs that have been created.

        :param filename: str, The JSON metamodel file path.
        """
        if not os.path.exists(filename):
            raise Exception("File does not exist: %s" % filename)

        self.filename = filename
        self.file = json.load(open(self.filename))

    def set_analysis(self, moniker):
        """
        Set the index of the analysis based on the ID or the name of the analysis.

        :param moniker: str, Analysis ID or name.
        :return: bool
        """
        for idx, analysis in enumerate(self.file):
            if analysis['name'] == moniker:
                self.set_i = idx

                return True

        raise Exception("Could not load the model: %s" % moniker)

    def downsamples(self, model_name):
        """
        Return the downsamples list from the metamodels.json file that was passed in.

        :param model_name: str, name of the model to look for down samples
        :return: list, Downsamples.
        """
        # check if the model_name has the downsamples, else, rely on the default set
        # of downsamples.
        ds = self.algorithm_options.get(model_name, {}).get('downsamples', None)
        if ds is None:
            ds = self.file[self.set_i].get('downsamples', None)
        return ds

    @property
    def results_file(self):
        """
        Path to the results file that is to be processed. This is a CSV file.

        :return: str, path.
        """
        if 'results_file' in self.file[self.set_i].keys():
            # fully qualify the path
            p = self.file[self.set_i]['results_file']
            if os.path.isabs(p):
                return p
            else:
                return os.path.abspath(os.path.join(os.path.dirname(self.filename), p))
        else:
            return f"post_process/{self.analysis_name}/simulation_results.csv"

    @property
    def analysis_name(self):
        """
        Return the analysis name from the metamodels.json file that was passed in.

        :return: str, Analysis name.
        """
        return self.file[self.set_i]['name']

    @property
    def algorithm_options(self):
        """
        Return the algorithm options from the metamodels.json file that was passed in.

        :return: dict, Algorithm options.
        """

        def _remove_comments(data):
            """
            This method recursively goes through a dict and removes any '_comments' keys.

            :param data: dict, Data.
            :return:
            """
            for k, v in data.items():
                if isinstance(v, dict):
                    data[k] = _remove_comments(v)

            if '_comments' in data.keys():
                del data['_comments']

            return data

        options = self.file[self.set_i].get('algorithm_options', None)
        # Remove all the _comments strings from the algorithm_options string
        return _remove_comments(options)

    @property
    def validation_id(self):
        """
        Return the validation ID from the metamodels.json file that was passed in.

        :return: str, Validation ID.
        """
        return self.file[self.set_i]['validation_datapoint_id']

    def model_paths(self, model_type, response, downsample=None, root_path=None):
        """
        Return the paths to the model to be loaded. This includes the scaler value if the
        model requires the data to scale the input.

        If the root path is provided, then that path will take precedent over the downsample
        and no values passed format.

        :param model_type: str, The type of reduced order model (e.g. RandomForest).
        :param response: str, The response (or model) to load (e.g. ETSOutletTemperature).
        :param downsample: float, The downsample value to load.  Defaults to None.
        :param root_path: If used, then it is the root path of the models. The models will be in subdirectories for each
        of the model_types.
        :return: list, [model_path, scaler_path].
        """
        if root_path:
            if downsample:
                model_path = "%s_%s/%s/models/%s.pkl" % (root_path, downsample, model_type, response)
                scaler_path = "%s_%s/%s/scalers.pkl" % (root_path, downsample, model_type)
            else:
                model_path = "%s/%s/models/%s.pkl" % (root_path, model_type, response)
                scaler_path = "%s/%s/scalers.pkl" % (root_path, model_type)
        elif downsample:
            model_path = "output/%s_%s/%s/models/%s.pkl" % (self.analysis_name, downsample, model_type, response)
            scaler_path = "output/%s_%s/%s/models/scalers.pkl" % (self.analysis_name, downsample, model_type)
        else:
            model_path = "output/%s/%s/models/%s.pkl" % (self.analysis_name, model_type, response)
            scaler_path = "output/%s/%s/models/scalers.pkl" % (self.analysis_name, model_type)

        return model_path, scaler_path

    def models_exist(self, model_type, models_to_load=None, downsample=None, root_path=None):
        """
        Check if the models exist, if not, then return false.

        :param model_type: str, The type of reduced order model (e.g. RandomForest).
        :param models_to_load: list, Name of responses to load.
        :param downsample: float, The downsample value to load.  Defaults to None.
        :param root_path: If used, then it is the root path of the models. The models will be in subdirectories for each
        of the model_types.
        :return: bool
        """
        if models_to_load is None:
            models_to_load = []

        self.rom_type = model_type

        if not models_to_load:
            models_to_load = self.available_response_names(self.rom_type)

        print("Checking if models exist %s" % models_to_load)
        exist = []
        for response in models_to_load:
            model_path, _ = self.model_paths(
                self.rom_type, response, downsample=downsample, root_path=root_path
            )
            exist.append(os.path.exists(model_path))

        return all(exist)

    def load_models(self, model_type, models_to_load=None, downsample=None, root_path=None):
        """
        Load in the metamodels/generators.

        :param model_type: str, The type of reduced order model (e.g. RandomForest).
        :param models_to_load: list, Name of responses to load.
        :param downsample: float, The downsample value to load.  Defaults to None.
        :return: dict, Metrics {response, model type, downsample, load time, disk size}.
        """
        if models_to_load is None:
            models_to_load = []

        self.rom_type = model_type

        if not models_to_load:
            models_to_load = self.available_response_names(self.rom_type)

        metrics = {'response': [], 'model_type': [], 'downsample': [],
                   'load_time': [], 'disk_size': []}
        for response in models_to_load:
            print("Loading %s model for response: %s" % (model_type, response))

            start = time.time()
            model_path, scaler_path = self.model_paths(
                self.rom_type, response, downsample=downsample, root_path=root_path
            )

            self.models[response] = ETSModel(response, model_path, scaler_path)
            metrics['response'].append(response)
            metrics['model_type'].append(model_type)
            metrics['downsample'].append(downsample)
            metrics['load_time'].append(time.time() - start)
            metrics['disk_size'].append(os.path.getsize(model_path))

        print("Finished loading models")
        print("The responses are:")
        for index, rs in enumerate(self.available_response_names(self.rom_type)):
            print("  %s: %s" % (index, rs))

        print("The covariates are:")
        for index, cv in enumerate(self.covariate_names(self.rom_type)):
            print("  %s: %s" % (index, cv))

        return metrics

    def yhats(self, data, prepend_name, response_names=None, ):
        """
        Run predict on multiple responses with the supplied data and store the results in the
        supplied DataFrame.

        The prepend_name is needed in order to not overwrite the existing data in the dataframe
        after evaluation. For example, if the response name is HeatingElectricity, the supplied
        data may already have that field provided; therefore, this method adds the prepend_name to
        the newly predicted data. If prepend_name is set to 'abc', then the new column would be
        'abc_HeatingElectricity'.

        :param data: pandas DataFrame, Values to predict on.
        :param prepend_name: str, Name to prepend to the beginning of each of the response names.
        :param response_names: list, Responses to evaluate. If None, then defaults to all the available_response_names.
        :return: pandas DataFrame, Original data with added predictions.
        """
        if not response_names:
            response_names = self.available_response_names(self.rom_type)

        # Verify that the prepend_name is not going to raise an exception
        colnames = data.columns.values
        for response_name in response_names:
            if f'{prepend_name}_{response_name}' in colnames:
                raise DuplicateColumnName(
                    f'{prepend_name}_{response_name} will result in duplicate. Set prepend_name to another value')

        for response_name in response_names:
            data[f"{prepend_name}_{response_name}"] = self.yhat(response_name, data)

        return data

    def yhat(self, response_name, data):
        """
        Run predict on the selected model (response) with the supplied data.

        :param response_name: str, Name of the model to evaluate.
        :param data: pandas DataFrame, Values to predict on.
        :return: pandas DataFrame, Predictions.
        :raises: Exception: Model does not have the response.
        """

        if response_name not in self.available_response_names(self.rom_type):
            raise Exception("Model does not have the response '%s'" % response_name)

        # Verify that the covariates are defined in the DataFrame, if not, then remove them before
        # calling the yhat method
        extra_columns_in_df = list(
            set(data.columns.values) - set(self.covariate_names(self.rom_type)))
        missing_data_in_df = list(
            set(self.covariate_names(self.rom_type)) - set(data.columns.values))

        if len(extra_columns_in_df) > 0:
            # print("Removing unneeded column before evaluation")
            data = data.drop(columns=extra_columns_in_df)

        if len(missing_data_in_df) > 0:
            print("Error: The following columns are missing in the DataFrame")
            raise Exception("Need to define %s in DataFrame for model" % missing_data_in_df)

        # Typecast the columns before running the analysis
        data[self.covariate_types(self.rom_type)['float']] = data[
            self.covariate_types(self.rom_type)['float']
        ].astype(float)
        data[self.covariate_types(self.rom_type)['int']] = data[
            self.covariate_types(self.rom_type)['int']
        ].astype(int)

        # Order the data columns correctly -- this is a magic function but is the order is
        # imperative when predicting.
        data = data[self.covariate_names(self.rom_type)]

        # Transform cyclical columns
        for cv in self.covariates(self.rom_type):
            if cv.get('algorithm_options', None):
                if cv['algorithm_options'].get(self.rom_type, None):
                    if cv['algorithm_options'][self.rom_type].get('variable_type', None):
                        if cv['algorithm_options'][self.rom_type]['variable_type'] == 'cyclical':
                            print("Transforming covariate to be cyclical %s" % cv['name'])
                            data[cv['name']] = data.apply(
                                apply_cyclic_transform,
                                column_name=cv['name'],
                                category_count=cv['algorithm_options'][self.rom_type][
                                    'category_count'],
                                axis=1
                            )

        return self.models[response_name].yhat(data)

    def save_csv(self, data, csv_name):
        """
        Save pandas DataFrame in CSV format.

        :param data: pandas DataFrame, Data to be exported.
        :param csv_name: str, Name of the CSV file.
        :return:
        """
        lookup_table_dir = 'output/%s/%s/lookup_tables/' % (
            self.analysis_name,
            self.rom_type
        )
        if not os.path.exists(lookup_table_dir):
            os.makedirs(lookup_table_dir)

        file_name = '%s/%s.csv' % (
            lookup_table_dir,
            csv_name)

        data.to_csv(file_name, index=False)

    def save_2d_csvs(self, data, first_dimension, file_prepend):
        # TODO: move this to a general helper location and remove the auto generation of the save path
        """
        Generate 2D (time, first) CSVs based on the model loaded and the two dimensions.

        The rows are the datetimes as defined in the data (DataFrame).

        :param data: pandas DataFrame
        :param first_dimension: str, The column heading variable.
        :param file_prepend: str, Special variable to prepend to the file name.
        :return: None
        """

        # Create the lookup table directory - probably want to make this a base class for all
        # python scripts that use the filestructure to store the data.
        lookup_table_dir = 'output/%s/%s/lookup_tables/' % (
            self.analysis_name,
            self.rom_type
        )
        if not os.path.exists(lookup_table_dir):
            os.makedirs(lookup_table_dir)

        for response in self.loaded_models:
            print("Creating CSV for %s" % response)

            # TODO: look into using DataFrame.pivot() to transform data
            file_name = '%s/%s_%s.csv' % (
                lookup_table_dir,
                file_prepend,
                response
            )

            # Save the data times in a new DataFrame (will be in order).
            save_df = pd.DataFrame.from_dict({'datetime': data['datetime'].unique()})
            for unique_value in data[first_dimension].unique():
                new_df = data[data[first_dimension] == unique_value]
                # add in the type of model
                if self.rom_type == 'RandomForest':
                    short_model_name = f'RF_{response}'
                else:
                    raise Exception("Need to create model lookup!")
                save_df[unique_value] = new_df[short_model_name].values

            save_df.to_csv(file_name, index=False)

    def save_3d_csvs(self, data, first_dimension, second_dimension, second_dimension_short_name,
                     file_prepend, save_figure=False):
        # TODO: move this to a general helper location and remove the auto generation of the save path
        """
        Generate 3D (time, first, second) CSVs based on the model loaded and the two dimensions.
        The second dimension becomes individual files.

        The rows are the datetimes as defined in the data (DataFrame)

        :param data: pandas DataFrame
        :param first_dimension: str, The column heading variable.
        :param second_dimension: str, The values that will be reported in the table.
        :param second_dimension_short_name: str, Short display name for second variable (for filename).
        :param file_prepend: str, Special variable to prepend to the file name.
        :return: None
        """

        # Create the lookup table directory - probably want to make this a base class for all
        # python scripts that use the filestructure to store the data.
        lookup_table_dir = 'output/%s/%s/lookup_tables/' % (
            self.analysis_name,
            self.rom_type
        )
        if not os.path.exists(lookup_table_dir):
            os.makedirs(lookup_table_dir)

        for response in self.loaded_models:
            print("Creating CSV for %s" % response)

            # TODO: look into using DataFrame.pivot() to transform data
            for unique_value in data[second_dimension].unique():
                file_name = '%s/%s_%s_%s_%.2f.csv' % (
                    lookup_table_dir,
                    file_prepend,
                    response,
                    second_dimension_short_name,
                    unique_value)
                lookup_df = data[data[second_dimension] == unique_value]

                # Save the data times in a new dataframe (will be in order)
                save_df = pd.DataFrame.from_dict({'datetime': lookup_df['datetime'].unique()})
                for unique_value_2 in data[first_dimension].unique():
                    new_df = lookup_df[lookup_df[first_dimension] == unique_value_2]
                    save_df[unique_value_2] = new_df[response].values

                save_df.to_csv(file_name, index=False)

                # Create heat maps
                if save_figure:
                    figure_filename = 'output/%s/%s/images/%s_%s_%s_%.2f.png' % (
                        self.analysis_name,
                        self.rom_type,
                        file_prepend,
                        response,
                        second_dimension_short_name,
                        unique_value)

                    # This is a bit cheezy right now, load in the file and process again
                    df_heatmap = pd.read_csv(file_name, header=0)

                    # Remove the datetime column before converting the column headers to rounded floats
                    df_heatmap = df_heatmap.drop(columns=['datetime'])
                    df_heatmap.rename(columns=lambda x: round(float(x), 1), inplace=True)

                    plt.figure()
                    f, ax = plt.subplots(figsize=(5, 12))
                    sns.heatmap(df_heatmap)
                    ax.set_title('%s - Mass Flow %s kg/s' % (response, unique_value))
                    ax.set_xlabel('ETS Inlet Temperature')
                    ax.set_ylabel('Hour of Year')
                    plt.savefig(figure_filename)
                    plt.close('all')

    def model(self, response_name):
        """
        Return model for specific response.

        :param response_name: str, Name of model response.
        """
        if response_name not in self.available_response_names(self.rom_type):
            raise Exception("Model does not have the response '%s'" % response_name)

        return self.models[response_name].model

    @property
    def loaded_models(self):
        """
        Return the list of available keys in the models dictionary.

        :return: list, Responses.
        """
        return self.models.keys()

    @property
    def analysis(self):
        """
        Return the metamodel analysis file.

        :return: Parsed JSON metamodel file.
        """
        if self.set_i is None:
            raise Exception(
                "Attempting to access analysis without setting. Run analysis.set_analysis(<id>)"
            )

        return self.file[self.set_i]

    def covariates(self, model_type):
        """
        Return dictionary of covariates for specified model type.

        :param model_type: str, The type of reduced order model (e.g. RandomForest).
        :return: dict, Covariates.
        """

        if self.set_i is None:
            raise Exception(
                "Attempting to access analysis without setting. Run analysis.set_analysis(<id>)"
            )

        # Only return the covariates that don't have ignore true for the type of model
        results = []
        for cv in self.file[self.set_i]['covariates']:
            if not cv.get('algorithm_options', {}).get(model_type, {}).get('ignore', False):
                results.append(cv)

        return results

    def covariate_types(self, model_type):
        """
        Return dictionary of covariate types.

        :param model_type: str, The type of reduced order model (e.g. RandomForest).
        :return: dict, {'type':['covariate name']}.
        """

        if self.set_i is None:
            raise Exception("Attempting to access analysis without setting. Run analysis.set_analysis(<id>)")

        # Group the datetypes by column
        data_types = {
            'float': [],
            'str': [],
            'int': []
        }
        for cv in self.covariates(model_type):
            data_types[cv['type']].append(cv['name'])

        return data_types

    def covariate_names(self, model_type):
        """
        Return a list of covariate names. The order in the JSON file must be the order that is passed into the
        metamodel, otherwise the data will not make sense.

        :param model_type: str, The type of reduced order model (e.g. RandomForest).
        :return: list, Covariate names.
        """

        if self.set_i is None:
            raise Exception(
                "Attempting to access analysis without setting. Run analysis.set_analysis(<id>)"
            )

        return [cv['name'] for cv in self.covariates(model_type)]

    def available_response_names(self, _model_type):
        """
        Return a list of response names.

        :param _model_type: str, The type of reduced order model (e.g. RandomForest).
        :return: list, Response names.
        """

        if self.set_i is None:
            raise Exception(
                "Attempting to access analysis without setting. Run analysis.set_analysis(<id>)"
            )

        return [cv['name'] for cv in self.file[self.set_i]['responses']]

    @classmethod
    def resolve_algorithm_options(cls, algorithm_options):
        """
        Go through the algorithm options that are in the metamodel.json file and run 'eval' on the strings.
        This allows complex strings to exist in the json file that get expanded as necessary.

        # TODO: Add an example

        :param algorithm_options: dict, the algorithm options to run eval on
        :return:
        """
        for k, v in algorithm_options.items():
            if isinstance(v, dict):
                algorithm_options[k] = Metamodels.resolve_algorithm_options(v)
            elif isinstance(v, str) and 'eval(' in v:
                # remove eval() from string in file and then call it
                string_value = re.search('eval\((.*)\)', v).groups()[0]
                algorithm_options[k] = eval(string_value)
        return algorithm_options
