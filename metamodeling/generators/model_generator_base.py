# -*- coding: utf-8 -*-
"""
.. moduleauthor:: Nicholas Long (nicholas.l.long@colorado.edu, nicholas.lee.long@gmail.com)
"""
import fnmatch
import os
import random
import shutil
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..shared import apply_cyclic_transform, pickle_file


class ModelGeneratorBase(object):
    def __init__(self, analysis_id, random_seed=None, **kwargs):
        """
        Base class for generating ROMs

        :param analysis_id: string, identifier of the model to build
        :param random_seed: int, random seed to use
        :param kwargs:
            See below

        :Keyword Arguments:
            * *downsample* (``double``) -- Fraction to downsample the dataframe. If this exists
              then the data will be downsampled, and the results will be stored in a directory with
              this value appended.
        """
        self.analysis_id = analysis_id
        self.random_seed = random_seed if random_seed else np.random.seed(time.time())
        self.model_results = []
        self.model_type = self.__class__.__name__
        self.dataset = None
        self.downsample = kwargs.get('downsample', None)

        print("Initializing %s" % self.model_type)

        # Initialize the directories where results are to be stored.
        if self.downsample:
            self.base_dir = 'output/%s_%s/%s' % (self.analysis_id, self.downsample, self.model_type)
        else:
            self.base_dir = 'output/%s/%s' % (self.analysis_id, self.model_type)
        self.images_dir = '%s/images' % self.base_dir
        self.models_dir = '%s/models' % self.base_dir
        if self.downsample:
            self.validation_dir = 'output/%s_%s/ValidationData' % (
                self.analysis_id, self.downsample)
        else:
            self.validation_dir = 'output/%s/ValidationData' % self.analysis_id
        self.data_dir = '%s/data' % self.base_dir

        # Remove some directories if they exist
        for dir_n in ['images_dir', 'models_dir']:
            if os.path.exists(getattr(self, dir_n)):
                # print("removing the directory %s" % dir)
                shutil.rmtree(getattr(self, dir_n))

        # create directory if not exist for each of the above
        for dir_n in ['base_dir', 'images_dir', 'models_dir', 'data_dir', 'validation_dir']:
            if not os.path.exists(getattr(self, dir_n)):
                os.makedirs(getattr(self, dir_n))

        for root, dirnames, filenames in os.walk(self.base_dir):
            for filename in fnmatch.filter(filenames, 'cv_results_*.csv'):
                os.remove('%s/%s' % (self.base_dir, filename))

            for filename in fnmatch.filter(filenames, 'model_results.csv'):
                os.remove('%s/%s' % (self.base_dir, filename))

    def save_dataframe(self, dataframe, path):
        pickle_file(dataframe, path)

    def inspect(self):
        """
        Inspect the dataframe and return the statistics of the dataframe.

        :return:
        """
        # look at the entire datatset and save the statistics from the file to the data_dir
        out_df = self.dataset.describe()
        out_df.to_csv(f'{self.data_dir}/statistics.csv')

        # list out all the columns
        out_df = self.dataset.columns
        with open(f'{self.data_dir}/column_names.csv', 'w') as f:
            for column in self.dataset.columns:
                f.write(column + '\n')

    def load_data(self, datafile):
        """
        Load the data into a dataframe. The data needs to be a CSV file at the moment.

        :param datafile: str, path to the CSV file to load
        :return: None
        """
        if os.path.exists(datafile):
            self.dataset = pd.read_csv(datafile)
        else:
            raise Exception(f"Datafile does not exist: {datafile}")

        print(f'Loading results data file: {datafile}')

    def evaluate(self, model, model_name, model_moniker, x_data, y_data, downsample,
                 build_time, cv_time, covariates=None, scaler=None):
        """
        Generic base function to evaluate the performance of the models.

        :param model:
        :param model_name:
        :param x_data:
        :param y_data:
        :param downsample:
        :param build_time:
        :return: Ordered dict
        """
        yhat = model.predict(x_data)

        if scaler:
            yhat = scaler[model_name].inverse_transform(yhat)
            y_data = scaler[model_name].inverse_transform(y_data)

        errors = abs(yhat - y_data)
        spearman = spearmanr(y_data, yhat)
        pearson = pearsonr(y_data, yhat)

        slope, intercept, r_value, _p_value, _std_err = stats.linregress(y_data, yhat)

        self.yy_plots(y_data, yhat, model_name)
        self.qq_plots(x_data, y_data, yhat, model_name)

        return yhat, OrderedDict([
            ('name', model_name),
            ('model_type', model_moniker),
            ('downsample', downsample),
            ('slope', slope),
            ('intercept', intercept),
            ('mae', np.mean(errors)),
            ('r_value', r_value),
            ('r_squared', r_value ** 2),
            ('spearman', spearman[0]),
            ('pearson', pearson[0]),
            ('time_to_build', build_time),
            ('time_to_cv', cv_time),
        ])

    def build(self, metamodel, **kwargs):
        if self.dataset is None:
            raise Exception("Need to load the datafile first by calling Metamodel.load_data(<path-to-file>)")

        # Type cast the columns - this is probably not needed
        data_types = metamodel.covariate_types(self.model_type)
        self.dataset[data_types['float']] = self.dataset[data_types['float']].astype(float)
        self.dataset[data_types['int']] = self.dataset[data_types['int']].astype(int)

    def train_test_validate_split(self, dataset, metamodel, downsample=None, scale=False):
        """
        Use the built in method to generate the train and test data. This adds an additional
        set of data for validation. This vaildation dataset is a unique ID that is pulled out
        of the dataset before the test_train method is called.
        """
        print("Initial dataset size is %s" % len(dataset))
        validate_id = None
        if metamodel.validation_id == 'first':
            # grab the first id in the dataset. This is non-ideal, but allow for rapid testing
            validate_id = dataset.iloc[0]['id']
        elif metamodel.validation_id == 'median':
            raise Exception('Median validation ID is not implemented')
            # look at all of the covariates and try to find the median value from them all
            # this method should be deterministic

            # Code below only works if the space is fully filled out and if only looking at variables
            # that are constant for the whole annual simulation.
            # closest_medians = dataset
            # for cv in metamodel.covariates(self.model_type):
            #     if cv.get('alogithm_option', None):
            #         if cv['algorithm_options'].get(self.model_type, None):
            #             if cv['algorithm_options'][self.model_type]['ignore']:
            #                 continue
            #     median = dataset[cv['name']].median()
            #     print(f'my median is {median}')
            #     closest_medians = closest_medians[closest_medians[cv['name']] == median]
            #     print(f'len of dataframe is {len(closest_medians)}')

        elif metamodel.validation_id == 'random':
            ids = dataset['id'].unique()
            validate_id = random.choice(ids)
        else:
            # assume that there is a validation id that has been passed
            validate_id = metamodel.validation_id

        if validate_id and validate_id in dataset['id'].unique():
            print('Extracting validation dataset and converting to date time')
            validate_xy = dataset[dataset['id'] == validate_id]

            # Covert the validation dataset datetime to actual datetime objects
            # validate_xy['DateTime'] = pd.to_datetime(dataset['DateTime'])
            #
            # Constrain to minute precision to make this method much faster
            validate_xy['DateTime'] = validate_xy['DateTime'].astype('datetime64[m]')

            dataset = dataset[dataset['id'] != validate_id]
        else:
            raise Exception(
                "Validation id does not exist in dataframe. ID was %s" % validate_id)

        if downsample:
            num_rows = int(len(dataset.index.values) * downsample)
            print("Downsampling dataframe by %s to %s rows" % (downsample, num_rows))
            dataset = dataset.sample(n=num_rows)

        for cv in metamodel.covariates(self.model_type):
            if cv.get('algorithm_options', None):
                if cv['algorithm_options'].get(self.model_type, None):
                    if cv['algorithm_options'][self.model_type].get('variable_type', None):
                        if cv['algorithm_options'][self.model_type]['variable_type'] == 'cyclical':
                            print("Transforming covariate to be cyclical %s" % cv['name'])
                            dataset[cv['name']] = dataset.apply(
                                apply_cyclic_transform,
                                column_name=cv['name'],
                                category_count=cv['algorithm_options'][self.model_type]['category_count'],
                                axis=1
                            )

        train_x, test_x, train_y, test_y = train_test_split(
            dataset[metamodel.covariate_names(self.model_type)],
            dataset[metamodel.available_response_names(self.model_type)],
            train_size=0.7,
            test_size=0.3,
            random_state=self.random_seed
        )

        # If scaling, then fit the scaler on the training data, then use the trained data
        # scalar to scale the test data.
        if scale:
            scalers = {'features': StandardScaler()}
            train_x[train_x.columns] = scalers['features'].fit_transform(train_x[train_x.columns])
            test_x[test_x.columns] = scalers['features'].transform(test_x[test_x.columns])

            for response in metamodel.available_response_names(self.model_type):
                scalers[response] = StandardScaler()
                train_y[response] = scalers[response].fit_transform(
                    train_y[response].values.reshape(-1, 1)
                )
                test_y[response] = scalers[response].transform(
                    test_y[response].values.reshape(-1, 1)
                )
        else:
            scalers = None

        print("Dataset size is %s" % len(dataset))
        print("Training dataset size is %s" % len(train_x))
        print("Validation dataset size is %s" % len(validate_xy))

        return train_x, test_x, train_y, test_y, validate_xy, scalers

    def yy_plots(self, y_data, yhat, model_name):
        """
        Plot the yy-plots

        :param y_data:
        :param yhat:
        :param model_name:
        :return:
        """
        # This need to be updated with the creating a figure with a size
        sns.set(color_codes=True)

        # Find the items that are zero / zero across y and yhat and remove to look at
        # plots and other statistics
        clean_data = zip(y_data, yhat)
        clean_data = [x for x in clean_data if x != (0, 0)]
        y_data = np.asarray([y[0] for y in clean_data])
        yhat = np.asarray([y[1] for y in clean_data])

        # Convert data to dataframe
        data = pd.DataFrame.from_dict({'Y': y_data, 'Yhat': yhat})

        with plt.rc_context(dict(sns.axes_style("whitegrid"))):
            fig = plt.figure(figsize=(6, 6), dpi=100)
            sns.regplot(
                x='Y',
                y='Yhat',
                data=data,
                ci=None,
                scatter_kws={"s": 50, "alpha": 1}
            )
            plt.tight_layout()
            plt.savefig('%s/fig_yy_%s.png' % (self.images_dir, model_name))
            fig.clf()
            plt.clf()
            plt.close('all')

        # Hex plots for YY data
        with plt.rc_context(dict(sns.axes_style("ticks"))):
            newplt = sns.jointplot(
                x=data['Y'], y=data['Yhat'], kind="hex", space=0
            )
            newplt.savefig('%s/fig_yy_hexplot_%s.png' % (self.images_dir, model_name))
            plt.clf()
            plt.close('all')

            # Remove 0,0 points for higher resolution
            sub_data = data[(data.Y != 0) & (data.Yhat != 0)]
            # Hex plots for YY data
            newplt = sns.jointplot(
                x=sub_data['Y'], y=sub_data['Yhat'], kind="hex", space=0
            )
            newplt.savefig('%s/fig_yy_hexplot_hres_%s.png' % (self.images_dir, model_name))
            plt.clf()
            plt.close('all')

    def qq_plots(self, x_data, y_data, y_hat, model_name):
        """Create QQ plots of the data.
        """
        # This need to be updated with the creating a figure with a size
        sns.set(color_codes=True)

        # save off all the data for later analysis
        single_df = x_data.copy()
        single_df['y'] = y_data
        single_df['y_hat'] = y_hat
        single_df['residuals'] = single_df['y'] - single_df['y_hat']
        # single_df.to_csv('%s/residuals_qq_%s.csv' % (self.data_dir, model_name))

        # Residual plots
        with plt.rc_context(dict(sns.axes_style("whitegrid"))):
            fig = plt.figure(figsize=(6, 6), dpi=100)
            sns.regplot(
                x='y',
                y='residuals',
                data=single_df,
                ci=None,
                scatter_kws={"s": 50, "alpha": 1}
            )
            plt.tight_layout()
            plt.savefig('%s/fig_residuals_%s.png' % (self.images_dir, model_name))
            fig.clf()
            plt.clf()
            plt.close('all')

        # QQ plots (todo)
