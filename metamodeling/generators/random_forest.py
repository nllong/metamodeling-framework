# -*- coding: utf-8 -*-
"""
.. moduleauthor:: Nicholas Long (nicholas.l.long@colorado.edu, nicholas.lee.long@gmail.com)
"""
import multiprocessing
import os
import time
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz

from .model_generator_base import ModelGeneratorBase
from ..shared import pickle_file, save_dict_to_csv, zipdir


class RandomForest(ModelGeneratorBase):
    def __init__(self, analysis_id, random_seed=None, **kwargs):
        super().__init__(analysis_id, random_seed, **kwargs)

    def export_tree_png(self, tree, covariates, filename):
        export_graphviz(tree, feature_names=np.asarray(covariates), filled=True, rounded=True)
        copy_tree_path = os.path.join(os.path.dirname(filename), 'tree.dot')
        os.rename('tree.dot', copy_tree_path)
        os.system('dot -Tpng %s -o %s' % (copy_tree_path, filename))
        if os.path.exists(copy_tree_path):
            os.remove(copy_tree_path)

    def evaluate(self, model, model_name, model_type, x_data, y_data, downsample,
                 build_time, cv_time, covariates=None, scaler=None):
        """
        Evaluate the performance of the forest based on known x_data and y_data.

        :param model:
        :param model_name:
        :param model_type:
        :param x_data:
        :param y_data:
        :param downsample:
        :param build_time:
        :param cv_time:
        :param covariates:
        :return:
        """
        _yhat, performance = super().evaluate(
            model, model_name, model_type, x_data, y_data, downsample,
            build_time, cv_time, covariates, scaler
        )

        importance_data = pd.Series(model.feature_importances_, index=np.asarray(covariates))
        importance_data = importance_data.nlargest(20)

        fig = plt.figure(figsize=(8, 4), dpi=100)
        # defaults to the ax in the figure.
        ax = sns.barplot(x=list(importance_data), y=list(importance_data.index.values),
                         color="grey", ci=None)
        # ax.set(xlabel='Relative Importance', ylabel='')
        ax.set_xlabel('Relative Importance')
        plt.tight_layout()
        fig.savefig('%s/fig_importance_%s.png' % (self.images_dir, model_name))
        fig.clf()
        plt.clf()

        # plot a single tree
        # TODO: add a configuration option on when to export the tree. This can take a long
        # time to export with large trees.
        # if downsample <= 0.01:
        #   tree_file_name = '%s/fig_first_tree_%s.png' % (self.images_dir, model_name)
        #   self.export_tree_png(model.estimators_[0], covariates, tree_file_name)

        # add some more data to the model evaluation dict
        performance['n_estimators'] = model.n_estimators
        performance['max_depth'] = model.max_depth if not model.max_depth else 0
        performance['max_features'] = model.max_features
        performance['min_samples_leaf'] = model.min_samples_leaf
        performance['min_samples_split'] = model.min_samples_leaf

        return performance

    def save_cv_results(self, cv_results, response, downsample, filename):
        """
        Save the cv_results to a CSV file. Data in the cv_results file looks like the following.

        The CV results are the results of the GridSearch k-fold cross validation. The form of the
        results take the following from:

        .. code-block:: python

            {
                'param_kernel': masked_array(data=['poly', 'poly', 'rbf', 'rbf'],
                                             mask=[False False False False]...)
                'param_gamma': masked_array(data=[-- -- 0.1 0.2],
                                            mask=[True  True False False]...),
                'param_degree': masked_array(data=[2.0 3.0 - - --],
                                             mask=[False False  True  True]...),
                'split0_test_score': [0.8, 0.7, 0.8, 0.9],
                'split1_test_score': [0.82, 0.5, 0.7, 0.78],
                'mean_test_score': [0.81, 0.60, 0.75, 0.82],
                'std_test_score': [0.02, 0.01, 0.03, 0.03],
                'rank_test_score': [2, 4, 3, 1],
                'split0_train_score': [0.8, 0.9, 0.7],
                'split1_train_score': [0.82, 0.5, 0.7],
                'mean_train_score': [0.81, 0.7, 0.7],
                'std_train_score': [0.03, 0.03, 0.04],
                'mean_fit_time': [0.73, 0.63, 0.43, 0.49],
                'std_fit_time': [0.01, 0.02, 0.01, 0.01],
                'mean_score_time': [0.007, 0.06, 0.04, 0.04],
                'std_score_time': [0.001, 0.002, 0.003, 0.005],
                'params': [{'kernel': 'poly', 'degree': 2}, ...],
            }

        :param cv_results:
        :param filename:
        :return:
        """
        data = {}
        data['downsample'] = []
        for params in cv_results['params']:
            for param, value in params.items():
                if not data.get(param, None):
                    data[param] = []
                data[param].append(value)
                data['downsample'] = downsample
                data['response'] = response
        data['mean_train_score'] = cv_results['mean_train_score']
        data['mean_test_score'] = cv_results['mean_test_score']
        data['mean_fit_time'] = cv_results['mean_fit_time']
        data['mean_score_time'] = cv_results['mean_score_time']
        data['rank_test_score'] = cv_results['rank_test_score']

        df = pd.DataFrame.from_dict(data)
        df.to_csv(filename)

    def build(self, metamodel, **kwargs):
        super().build(metamodel, **kwargs)

        analysis_options = kwargs.get('algorithm_options', {})

        train_x, test_x, train_y, test_y, validate_xy, _scaler = self.train_test_validate_split(
            self.dataset,
            metamodel,
            downsample=self.downsample
        )

        # Save the validate dataframe to be used later to validate the accuracy of the models
        self.save_dataframe(validate_xy, "%s/rf_validation" % self.validation_dir)

        for response in metamodel.available_response_names(self.model_type):
            print("Fitting random forest model for %s" % response)

            start = time.time()
            base_fit_params = analysis_options.get('base_fit_params', {})
            rf = RandomForestRegressor(**base_fit_params)
            base_model = rf.fit(train_x, train_y[response])
            build_time = time.time() - start

            # Evaluate the forest when building them
            self.model_results.append(
                self.evaluate(
                    base_model, response, 'base', test_x, test_y[response],
                    self.downsample, build_time, 0,
                    covariates=metamodel.covariate_names(self.model_type),
                    scaler=_scaler
                )
            )

            if not kwargs.get('skip_cv', False):
                rf = RandomForestRegressor()

                kfold = 3
                print('Perfoming CV with k-fold equal to %s' % kfold)
                # Grab the param grid from what was specified in the metamodels.json file
                param_grid = analysis_options.get('param_grid', {})
                total_candidates = 1
                for param, options in param_grid.items():
                    total_candidates = len(options) * total_candidates

                print('CV will result in %s candidates' % total_candidates)

                # Need to update to handle strings:
                # https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py  # noqa

                # Allow for the computer to be responsive during grid_search
                n_jobs = multiprocessing.cpu_count() - 1
                grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kfold,
                                           verbose=2, refit=True, n_jobs=n_jobs)

                start = time.time()
                grid_search.fit(train_x, train_y[response])
                cv_time = time.time() - start

                # This should work, but the size of the model is really large after, so
                # trying to recreate the best_rf to save space.
                # best_rf = grid_search.best_estimator_

                print('The best params were %s' % grid_search.best_params_)

                # Rebuild only the best rf, and save the results
                model = RandomForestRegressor(**grid_search.best_params_)
                best_model = model.fit(train_x, train_y[response])

                pickle_file(best_model, '%s/%s' % (self.models_dir, response))

                # Save the cv results
                self.save_cv_results(
                    grid_search.cv_results_, response, self.downsample,
                    '%s/cv_results_%s.csv' % (self.base_dir, response)
                )

                self.model_results.append(
                    self.evaluate(
                        best_model, response, 'best', test_x, test_y[response],
                        self.downsample, build_time, cv_time,
                        covariates=metamodel.covariate_names(self.model_type),
                        scaler=_scaler
                    )
                )
            else:
                pickle_file(base_model, '%s/%s' % (self.models_dir, response))

        if self.model_results:
            save_dict_to_csv(self.model_results, '%s/model_results.csv' % self.base_dir)

        # Zip up the models
        zipf = zipfile.ZipFile(
            '%s/models.zip' % self.models_dir, 'w', zipfile.ZIP_DEFLATED, allowZip64=True
        )
        zipdir(self.models_dir, zipf, '.pkl')
        zipf.close()

        # Save the data that was used in the models for future processing and analysis
        self.dataset.to_csv('%s/data.csv' % self.data_dir)
