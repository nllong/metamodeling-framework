# -*- coding: utf-8 -*-
"""
Process an EPW file

.. moduleauthor:: Nicholas Long (nicholas.l.long@colorado.edu, nicholas.lee.long@gmail.com)
"""
import csv
from collections import OrderedDict

import pandas as pd
from datetime import datetime


class EpwFile:
    def __init__(self, filepath):
        """
        Load an EPW file into memory

        :param filepath: String, Path to the file
        """

        self.data = []

        self.columns = [
            {
                'name': 'year',
                'long_name': 'year',
                'type': 'int',
                'units': '',
            }, {
                'name': 'month',
                'long_name': 'month',
                'type': 'int',
                'units': '',
            }, {
                'name': 'day',
                'long_name': 'day',
                'type': 'int',
                'units': '',
            }, {
                'name': 'hour',
                'long_name': 'hour',
                'type': 'int',
                'units': '',
            }, {
                'name': 'minute',
                'long_name': 'minute',
                'type': 'int',
                'units': '',
            }, {
                'name': 'data_source',
                'long_name': 'data_source',
                'type': 'str',
                'units': '',
            }, {
                'name': 'dry_bulb',
                'long_name': 'dry_bulb',
                'type': 'float',
                'units': 'deg C',
            }, {
                'name': 'dew_point',
                'long_name': 'dew_point',
                'type': 'float',
                'units': 'deg C',
            }, {
                'name': 'rh',
                'long_name': 'rh',
                'type': 'float',
                'units': 'percent',
            }

        ]
        self.column_names = [c['name'] for c in self.columns]
        self.start_day_of_week = '0'  # Sunday

        # '', '', 'atmos_pressure', 'ext_horz_rad', 'ext_dir_rad',
        # 'horz_ir_sky', 'glo_horz_rad', 'dir_norm_rad', 'dif_horz_rad',
        # 'glo_horz_illum', 'dir_norm_illum', 'dif_horz_illum', 'zen_lum', 'wind_dir',
        # 'wind_spd', 'tot_sky_cvr', 'opaq_sky_cvr', 'visibility', 'ceiling_hgt',
        # 'pres_weath_obs', 'pres_weath_codes', 'precip_wtr', 'aerosol_opt_depth',
        # 'snow_depth', ' days_since_last_snow', 'albedo', 'rain', 'rain_quantity']
        # Date,HH:MM,Datasource,DryBulb {C},DewPoint {C},RelHum {%},Atmos Pressure {Pa},ExtHorzRad {Wh/m2},ExtDirRad {Wh/m2},HorzIRSky {Wh/m2},GloHorzRad {Wh/m2},DirNormRad {Wh/m2},DifHorzRad {Wh/m2},GloHorzIllum {lux},DirNormIllum {lux},DifHorzIllum {lux},ZenLum {Cd/m2},WindDir {deg},WindSpd {m/s},TotSkyCvr {.1},OpaqSkyCvr {.1},Visibility {km},Ceiling Hgt {m},PresWeathObs,PresWeathCodes,Precip Wtr {mm},Aerosol Opt Depth {.001},SnowDepth {cm},Days Last Snow,Albedo {.01},Rain {mm},Rain Quantity {hr}
        with open(filepath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == 'LOCATION':
                    pass
                    # print('Parsing Location')
                elif row[0] == 'DESIGN CONDITIONS':
                    pass
                    # print('Parsing Design Conditions')
                elif row[0] == 'TYPICAL/EXTREME PERIODS':
                    pass
                    # print('Parsing Typical / Extreme Periods')
                elif row[0] == 'GROUND TEMPERATURES':
                    pass
                    # print('Parsing Ground Temperatures')
                elif row[0] == 'HOLIDAYS/DAYLIGHT SAVINGS':
                    pass
                    # print('Parsing Holidays / Daylight Savings')
                elif row[0] == 'COMMENTS 1':
                    pass
                    # print('Parsing Comments 1')
                elif row[0] == 'COMMENTS 2':
                    pass
                    # print('Parsing Comments 2')
                elif row[0] == 'DATA PERIODS':
                    pass
                    # print('Parsing Data Periods')
                else:
                    self._append_row(row)

        self.post_process_data()

    def _append_row(self, row):
        data_types = [c['type'] for c in self.columns]

        row = [eval("%s(\'%s\')" % cast) for cast in zip(data_types, row)]
        self.data.append(OrderedDict(zip(self.column_names, row)))

    def post_process_data(self):
        """
        Add in derived columns

        :return:
        """
        for index, datum in enumerate(self.data):
            dt = "%s/%s/2017 %s:00" % (datum['month'], datum['day'], datum['hour'] - 1)
            self.data[index]['datetime'] = dt

            # convert to dt
            dt_obj = datetime.strptime(dt, '%m/%d/%Y %H:%M')

            # Add in the DayOfYear for convenience.
            self.data[index]['DayOfYear'] = dt_obj.strftime('%j')

            # add in the day of the week
            self.data[index]['dayofweek'] = dt_obj.strftime('%A')
            self.data[index]['dayofweek_int'] = dt_obj.strftime('%w')  # 0 = sunday, 1 = monday, ...


    def as_dataframe(self):
        """
        Return the EPW file as a dataframe. This drops the data_source column for brevity.

        :return: pandas DataFrame
        """
        return pd.DataFrame(self.data).drop(columns=['data_source'])


if __name__ == '__main__':
    epw = EpwFile("USA_CO_Golden-NREL.724666_TMY3.epw")

    for index, datum in enumerate(epw.data):
        if index < 10:
            print(datum)
