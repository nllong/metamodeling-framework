# -*- coding: utf-8 -*-
"""
.. moduleauthor:: Nicholas Long (nicholas.l.long@colorado.edu, nicholas.lee.long@gmail.com)
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="ticks", color_codes=True)


def evaluate_process_cv_results(cv_result_file, response, output_dir):
    if os.path.exists(cv_result_file):
        # Load the cv results
        print("Reading CV results file: %s" % cv_result_file)
        df = pd.read_csv(cv_result_file)
        df = df.drop('response', 1)
        # Fill in the max_depth that has NA when it was set to auto
        df = df.fillna(0)
        df = df.drop('downsample', 1)
        df = df.drop('mean_score_time', 1)
        df = df.drop('rank_test_score', 1)
        df = df.drop('mean_train_score', 1)
        # df = df.drop('max_depth', 1)
        df = df.drop(df.columns[[0]], axis=1)
        newplt = sns.pairplot(df)
        newplt.savefig('%s/fig_cv_%s_pairplot.png' % (output_dir, response))
        plt.close('all')

        # Plot specific xy plots
        f, ax = plt.subplots(figsize=(6.5, 6.5))
        sns.despine(f, left=True, bottom=True)
        newplt = sns.jointplot(
            x=df['mean_fit_time'], y=df['mean_test_score'], kind="hex"
        ).set_axis_labels('Mean Fit Time (seconds)', 'Mean Test Score (fraction)')
        newplt.savefig('%s/fig_cv_%s_time_v_score_hex.png' % (output_dir, response))
        plt.close('all')

        # Plot specific xy plots -- darkgrid background
        with plt.rc_context(dict(sns.axes_style("whitegrid"))):
            f, ax = plt.subplots(figsize=(6.5, 6.5))
            newplt = sns.scatterplot(x=df['mean_fit_time'], y=df['mean_test_score'],
                                     ax=ax).get_figure()
            ax.set_xlabel('Mean Fit Time (seconds)')
            ax.set_ylabel('Mean Test Score (fraction)')
            newplt.savefig('%s/fig_cv_%s_time_v_score.png' % (output_dir, response))
            plt.close('all')


def evaluate_process_model_results(model_results_file, output_dir):
    if os.path.exists(model_results_file):
        # Process the model results
        df = pd.read_csv(model_results_file)
        # If best exists, then use that, otherwise, just use what is in the column
        if 'best' in df.model_type.unique():
            df = df[df.model_type == 'best']
        # If there are two similar columns then remove one of them and update the name of the remaining item
        if all(x in ['ETSHeatingOutletTemperature', 'ETSCoolingOutletTemperature'] for x in df.name.unique()):
            df = df[df.name != 'ETSCoolingOutletTemperature']
            df.loc[df.name == 'ETSHeatingOutletTemperature', 'name'] = 'ETSOutletTemperature'

        # Melt the data for plot purposes
        melted_df = pd.melt(
            df[['name', 'time_to_build', 'time_to_cv']],
            id_vars='name',
            var_name='model',
            value_name='time'
        )

        # Plot the data
        fig = plt.figure(figsize=(8, 3), dpi=100)
        # Defaults to the ax in the figure
        ax = sns.barplot(x='time', y='name', hue='model', data=melted_df, ci=None)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('')
        plt.tight_layout()
        fig.savefig('%s/fig_time_to_build.png' % output_dir)
        plt.close('all')


def evaluate_process_all_model_results(data, validation_dir):
    # For unique_value in data['name'].unique():
    sub_df = data[data['model_type'] == 'best'].sort_values(by=['name', 'model_method'])

    data.to_csv('%s/all_model_results.csv' % validation_dir, index=False)

    keep_cols = ['name', 'model_method', 'pearson']
    sub_df[keep_cols].to_csv('%s/pcc_model_results.csv' % validation_dir, index=False)
