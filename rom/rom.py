#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry point for building, evaluating, and validating reduced order models. Presently, this
file uses argparse to process the command line options and, in the same file, processes the
arguments. Ideally, the processing portion of this file is removed from the parsing. There is a
helper rom-runner.py file as a helper file at the root of the repository that passed
the arguments to this file.

.. moduleauthor:: Nicholas Long (nicholas.l.long@colorado.edu, nicholas.lee.long@gmail.com)
"""
import argparse
import os
import shutil
import time

import pandas as pd
from pyfiglet import Figlet

from .analysis_definition.analysis_definition import AnalysisDefinition
from .evaluate_helpers import (
    evaluate_process_model_results,
    evaluate_process_all_model_results,
    evaluate_process_cv_results,
)
from .metamodels import Metamodels
from .shared import unpickle_file
from .validation_helpers import validate_dataframe, validation_save_metrics
# from pprint import pprint as pp

# Make sure to keep these models here, optimizing imports will remove these
from .generators.linear_model import LinearModel  # noqa
from .generators.random_forest import RandomForest # noqa
from .generators.svr import SVR # noqa
# End include

# TODO: move this to a generic location and use in other locations (e.g. save_2d/3d).
NAMEMAP = {
    'LinearModel': 'LM',
    'RandomForest': 'RF',
    'SVR': 'SVR',
}
f = Figlet()
print(f.renderText('ROM Framework'))

parser = argparse.ArgumentParser()
parser.add_argument('action', default=None, choices=['inspect', 'build', 'evaluate', 'validate', 'run'])
parser.add_argument('-f', '--file', help='Metadata file to use', default='metamodels.json')
parser.add_argument('-a', '--analysis-moniker', help='Name of the Analysis Model', required=True)
parser.add_argument('-m', '--model-type', nargs='*',
                    choices=['LinearModel', 'RandomForest', 'SVR'],
                    default=['LinearModel', 'RandomForest', 'SVR'],
                    help='Type of model to build')

# Run file options
parser.add_argument('-ad', '--analysis-definition',
                    help='Definition of an analysis to run using the ROMs', default=None)
parser.add_argument('-w', '--weather', help='Weather file to run analysis-definition', default=None)
parser.add_argument('-o', '--output', help='File to save the results to', default=None)
downsample = parser.add_argument(
    '-d', '--downsample', default=None, type=float, help='Specific down sample value')
args = parser.parse_args()

print('Passed build_models.py args: %s' % args)
print('Loading definition file: %s' % args.file)
metamodel = Metamodels(args.file)

if metamodel.set_analysis(args.analysis_moniker):
    if args.action in ['inspect', 'build', 'evaluate']:
        all_model_results = {}
        for model_name in args.model_type:
            # Check if the model name has any downsampling override values
            algo_options = metamodel.algorithm_options.get(model_name, {})
            algo_options = Metamodels.resolve_algorithm_options(algo_options)
            downsamples = metamodel.downsamples(model_name)
            if args.downsample:
                if args.downsample not in metamodel.downsamples(model_name):
                    print("Downsample argument must exist in the downsample list in the JSON, remove downsample or add to list in metamodels.json")
                    exit(1)
                else:
                    downsamples = [args.downsample]
            print("Running %s model '%s' with downsamples '%s'" % (args.action, model_name, downsamples))

            for downsample in downsamples:
                if args.downsample and args.downsample != downsample:
                    continue

                if args.action in ['inspect', 'build']:
                    # for these two actions we need to load the data into the dataframe.
                    klass = globals()[model_name]
                    # Set the random seed so that the test libraries are the same across the
                    # models
                    model = klass(metamodel.analysis_name, 79, downsample=downsample)
                    model.load_data(metamodel.results_file)
                    if args.action == 'inspect':
                        # this will
                        model.inspect()
                    elif args.action == 'build':
                        model.build(
                            metamodel,
                            algorithm_options=algo_options,
                            skip_cv=downsample > 0.5
                        )
                elif args.action == 'evaluate':
                    base_dir_ds = "output/%s_%s/%s" % (
                        args.analysis_moniker, downsample, model_name)

                    output_dir = "%s/images/cv_results" % base_dir_ds
                    if os.path.exists(output_dir):
                        shutil.rmtree(output_dir)
                    os.makedirs(output_dir)

                    # Process the model results
                    model_results_file = '%s/model_results.csv' % base_dir_ds
                    evaluate_process_model_results(model_results_file, output_dir)

                    # If this is the first file, then read it into the all_model_results to
                    # create a DataFrame to add all the model results together
                    if str(downsample) not in all_model_results.keys():
                        if os.path.exists(model_results_file):
                            all_model_results[str(downsample)] = pd.read_csv(model_results_file)
                            all_model_results[str(downsample)]['model_method'] = model_name
                    else:
                        if os.path.exists(model_results_file):
                            new_df = pd.read_csv(model_results_file)
                            new_df['model_method'] = model_name
                            all_model_results[str(downsample)] = pd.concat(
                                [all_model_results[str(downsample)], new_df],
                                axis=0,
                                ignore_index=True,
                                sort=False
                            )

                    for response in metamodel.available_response_names(model_name):
                        # Process the CV results
                        cv_result_file = '%s/cv_results_%s.csv' % (base_dir_ds, response)
                        evaluate_process_cv_results(cv_result_file, response, output_dir)

        # Below are some options that require all the models to be processed before running
        if args.action == 'evaluate':
            # Save any combined datasets for the downsampled instance
            for index, data in all_model_results.items():
                if data.shape[0] > 0:
                    # Combine all the model results together and evaluate the results
                    validation_dir = "output/%s_%s/ValidationData/evaluation_images" % (
                        args.analysis_moniker, index
                    )
                    if os.path.exists(validation_dir):
                        shutil.rmtree(validation_dir)
                    os.makedirs(validation_dir)

                    evaluate_process_all_model_results(data, validation_dir)
    elif args.action == 'validate':
        # Validate requires iterating over the downsamples before the models
        if args.downsample and args.downsample not in metamodel.downsamples:
            print("Downsample argument must exist in the downsample list in the JSON")
            exit(1)

        for downsample in metamodel.downsamples(None):
            if args.downsample and args.downsample != downsample:
                continue

            validation_dir = "output/%s_%s/ValidationData" % (args.analysis_moniker, downsample)
            output_dir = "%s/images" % validation_dir

            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)

            # VALIDATE MODELS - load the model into the Metamodel class. Seems like we can simplify
            # this to have the two classes rely on each other.
            metadata = {}

            # List of response files to load based on priority. The SVR models have an additional
            # variable that is needed, so prioritize that one
            preferred_validation_data = [
                "%s/%s" % (validation_dir, 'svr_validation.pkl'),
                "%s/%s" % (validation_dir, 'rf_validation.pkl'),
                "%s/%s" % (validation_dir, 'lm_validation.pkl'),
            ]
            for f in preferred_validation_data:
                if os.path.exists(f):
                    print('Loading validation data from %s' % f)
                    validation_df = unpickle_file(f)
                    break
            else:
                validation_df = None

            models = [(m, NAMEMAP[m]) for m in args.model_type]

            # Dict to store the load time results
            metrics = {'response': [], 'model_type': [], 'downsample': [], 'load_time': [],
                       'disk_size': [], 'run_time_single': [], 'run_time_8760': []}
            for model_type in models:
                metadata[model_type[0]] = {'responses': [], 'moniker': model_type[1]}

                if metamodel.models_exist(model_type[0], downsample=downsample):
                    ind_metrics = metamodel.load_models(model_type[0], downsample=downsample)
                    for item, values in ind_metrics.items():
                        metrics[item] = metrics[item] + ind_metrics[item]

                    # Run the ROM for each of the response variables
                    for response in metamodel.available_response_names(model_type[0]):
                        metadata[model_type[0]]['responses'].append(response)

                        start = time.time()
                        var_name = "Modeled %s %s" % (model_type[1], response)
                        validation_df[var_name] = metamodel.yhat(response, validation_df)
                        metrics['run_time_8760'].append(time.time() - start)

                        # Grab a single row for performance benchmarking
                        single_row = validation_df.iloc[[5]]
                        start = time.time()
                        metamodel.yhat(response, single_row)
                        metrics['run_time_single'].append(time.time() - start)
                else:
                    print("Persisted models for %s:%s do not exist" % (model_type[0], downsample))

            # Save the model performance data
            validation_save_metrics(pd.DataFrame.from_dict(metrics), output_dir)

            # Run bunch of validations on the loaded models
            validate_dataframe(validation_df, metadata, output_dir)
    elif args.action == 'run':
        print("Running")
        if not args.downsample:
            print("Must supply at least one downsample when running ROM models")
            exit(1)
        elif not args.analysis_definition:
            print("Must supply analysis definition when running ROM models")
            exit(1)

        analysis = AnalysisDefinition(args.analysis_definition)

        # Load the weather data if it exists
        if args.weather:
            analysis.load_weather_file(args.weather)
        data = analysis.as_dataframe()

        # Get a list of models with short names
        metadata = {}
        models = [(m, NAMEMAP[m]) for m in args.model_type]
        for model in models:
            metadata[model[0]] = {'responses': [], 'moniker': model[1]}

            # Load the reducted order models
            metamodel.load_models(model[0], downsample=args.downsample)

            # Run the ROM for each of the response variables
            for response in metamodel.available_response_names(model[0]):
                metadata[model[0]]['responses'].append(response)

                var_name = "Modeled %s %s" % (model[1], response)
                data[var_name] = metamodel.yhat(response, data)

        print(data.describe())
        if args.output:
            data.to_csv(args.output)
