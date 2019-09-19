# -*- coding: utf-8 -*-
"""
Parser for analysis definition JSON files.

.. moduleauthor:: Nicholas Long (nicholas.l.long@colorado.edu, nicholas.lee.long@gmail.com)
"""

import json
import os

import numpy as np
import pandas as pd

from .epw_file import EpwFile


# {
#   "variables": [
#     {
#       "name": "Month",
#       "data_source": "weather_file"
#     },
#     {
#       "name": "ETSInletTemperature",
#       "data_source": "distribution",
#       "distribution": {
#         "minimum": 15,
#         "maximum": 25,
#         "number_of_samples": 10
#       }
#     },
#     {
#       "name": "DistrictHeatingMassFlowRate",
#       "data_source": "value",
#       "value": 0.5
#     },
#     {
#       "name": "DistrictCoolingMassFlowRate",
#       "data_source": "distribution",
#       "distribution": {
#         "minimum": 0.02,
#         "maximum": 4,
#         "number_of_samples": 10
#       }
#     },
#     {
#       "name": "SomeOtherField",
#       "data_source": "values",
#       "values": [0.5, 0.75, 1.0]
#     },
#   ]
# }

class AnalysisDefinition:
    """
    Pass in a definition file and a weather file to generate distributions of models
    """

    def __init__(self, definition_file):
        self.filename = None
        self.file = None
        self.analyses = []
        self.set_i = None
        self.weather_data = None

        self.load_files(definition_file)

    def load_files(self, definition_file):
        if not os.path.exists(definition_file):
            raise Exception("File does not exist: %s" % definition_file)

        self.filename = definition_file
        self.file = json.load(open(self.filename))

    def load_weather_file(self, weather_file):
        """
        Load in the weather file and convert the field names to what is expected in the
        JSON file
        :return:
        """
        if not os.path.exists(weather_file):
            raise Exception("Weather file does not exist: %s" % weather_file)

        epw_file = EpwFile(weather_file)
        self.weather_data = epw_file.as_dataframe()
        for variable in self.file['variables']:
            if variable['data_source'] == 'epw':
                # Rename the weather file fields to the ones defined in the JSON file
                self.weather_data = self.weather_data.rename(
                    columns={variable['data_source_field']: variable['name']}
                )

    def as_dataframe(self):
        """
        Return the dataframe with all the data needed to run the analysis defined in the
        json file.

        Note that the first field in the analysis definition json file must be a value or an EPW.

        :return: pandas dataframe
        """
        # Check if there is a epw file field
        # {dtype('int64'): Index([u'Month', u'Hour', u'DayofWeek', u'SiteOutdoorAirRelativeHumidity'],
        #                        dtype='object'),
        #  dtype('float64'): Index([u'SiteOutdoorAirDrybulbTemperature', u'ETSInletTemperature',
        #                           u'DistrictHeatingMassFlowRate', u'DistrictCoolingMassFlowRate'],
        #                          dtype='object')}
        use_epw = False
        for variable in self.file['variables']:
            if variable['data_source'] == 'epw':
                use_epw = True
                break

        seed_df = None
        if use_epw:
            seed_df = self.weather_data

        # Add in the static variables
        for variable in self.file['variables']:
            if variable['data_source'] == 'value':
                if seed_df is None:
                    seed_df = pd.DataFrame.from_dict({variable['name']: [variable['value']]})
                else:
                    seed_df[variable['name']] = variable['value']

        # Now add in the combinitorials
        for variable in self.file['variables']:
            if variable['data_source'] == 'distribution':
                df_to_append = seed_df.copy(deep=True)
                values = np.linspace(
                    variable['distribution']['minimum'],
                    variable['distribution']['maximum'],
                    variable['distribution']['number_of_samples']
                ).tolist()

                for index, value in enumerate(values):
                    if index == 0:
                        # first time through add the variable to seed_df, no need to append
                        seed_df[variable['name']] = value
                    else:
                        df_to_append[variable['name']] = value
                        seed_df = pd.concat([seed_df, df_to_append])

            if variable['data_source'] == 'values':
                df_to_append = seed_df.copy(deep=True)

                for index, value in enumerate(variable['values']):
                    if index == 0:
                        # first time through add the variable to seed_df, no need to append
                        seed_df[variable['name']] = value
                    else:
                        df_to_append[variable['name']] = value
                        seed_df = pd.concat([seed_df, df_to_append])

        return seed_df
