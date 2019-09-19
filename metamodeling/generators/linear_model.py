# -*- coding: utf-8 -*-
"""
.. moduleauthor:: Nicholas Long (nicholas.l.long@colorado.edu, nicholas.lee.long@gmail.com)
"""
import time
import zipfile

from ..shared import pickle_file, save_dict_to_csv, zipdir
from sklearn.linear_model import LinearRegression

from .model_generator_base import ModelGeneratorBase


class LinearModel(ModelGeneratorBase):
    def __init__(self, analysis_id, random_seed=None, **kwargs):
        super().__init__(analysis_id, random_seed, **kwargs)

    def evaluate(self, model, model_name, model_type, x_data, y_data, downsample,
                 build_time, cv_time, covariates=None, scaler=None):
        """
        Evaluate the performance of the forest based on known x_data and y_data. If the
        model was scaled, then the test data will already be scaled.
        """
        yhat, performance = super().evaluate(
            model, model_name, model_type, x_data, y_data, downsample,
            build_time, cv_time, covariates, scaler
        )
        self.anova_plots(y_data, yhat, model_name)
        return performance

    def build(self, metamodel, **kwargs):
        super().build(metamodel, **kwargs)

        # analysis_options = kwargs.get('algorithm_options', {})

        train_x, test_x, train_y, test_y, validate_xy, _scaler = self.train_test_validate_split(
            self.dataset,
            metamodel,
            downsample=self.downsample
        )

        # save the validate dataframe to be used later to validate the accuracy of the models
        self.save_dataframe(validate_xy, "%s/lm_validation" % self.validation_dir)

        for response in metamodel.available_response_names(self.model_type):
            print("Fitting Linear Model for %s" % response)
            trained_model = LinearRegression()

            start = time.time()
            trained_model.fit(train_x, train_y[response])
            build_time = time.time() - start

            pickle_file(trained_model, '%s/%s' % (self.models_dir, response))

            self.model_results.append(
                self.evaluate(
                    trained_model, response, 'best', test_x, test_y[response], self.downsample,
                    build_time, 0
                )
            )

        if self.model_results:
            save_dict_to_csv(self.model_results, '%s/model_results.csv' % self.base_dir)

        # zip up the models
        zipf = zipfile.ZipFile(
            '%s/models.zip' % self.models_dir, 'w', zipfile.ZIP_DEFLATED, allowZip64=True
        )
        zipdir(self.models_dir, zipf, '.pkl')
        zipf.close()

        # save the data that was used in the models for future processing and analysis
        self.dataset.to_csv('%s/data.csv' % self.data_dir)
