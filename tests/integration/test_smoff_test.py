# -*- coding: utf-8 -*-
"""
.. moduleauthor:: Nicholas Long (nicholas.l.long@colorado.edu, nicholas.lee.long@gmail.com)
"""

import os
from shutil import rmtree
from unittest import TestCase

import pytest

class TestRunningSmallOfficeTest(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.output_dirs = ['output/smoff_test_0.15', 'output/smoff_test_1.0']
        cls.models = ['LinearModel', 'RandomForest', 'SVR']

        for d in cls.output_dirs:
            if os.path.exists(d):
                rmtree(d)

    @pytest.mark.run(order=1)
    def test_inspect(self):
        cmd = 'metamodel.py inspect -f tests/integration/data/smoff.json -a smoff_test'
        os.system(cmd)
        # verify the results exist
        for d in self.output_dirs:
            for m in self.models:
                if d == 'output/smoff_test_1.0' and m == 'SVR':
                    # this model isn't created (takes too long).
                    continue

                check_path = f'{d}/{m}/data/statistics.csv'
                self.assertTrue(os.path.exists(check_path), msg=f'Could not find {check_path}')

    @pytest.mark.run(order=2)
    def test_build(self):
        # call via the command line interface
        cmd = 'metamodel.py build -f tests/integration/data/smoff.json -a smoff_test'
        os.system(cmd)

        files_to_check = [
            'images/fig_yy_HeatingElectricity.png',
            'images/fig_yy_hexplot_HeatingElectricity.png',
            'models/HeatingElectricity.pkl',
            'models/models.zip',
        ]
        for d in self.output_dirs:
            for m in self.models:
                if d == 'output/smoff_test_1.0' and m == 'SVR':
                    # this model isn't created (takes too long).
                    continue

                for f in files_to_check:
                    check_path = f'{d}/{m}/{f}'
                    self.assertTrue(os.path.exists(check_path), msg=f'Could not find {check_path}')

                # random spot check for the validation files
                check_path = f'{d}/ValidationData/lm_validation.pkl'
                self.assertTrue(os.path.exists(check_path), msg=f'Could not find {check_path}')

    @pytest.mark.run(order=3)
    def test_evaluate(self):
        cmd = 'metamodel.py evaluate -f tests/integration/data/smoff.json -a smoff_test'
        os.system(cmd)

        files_to_check = [
            'all_model_results.csv',
            'pcc_model_results.csv',
        ]

        for d in self.output_dirs:
            for f in files_to_check:
                check_path = f'{d}/ValidationData/evaluation_images/{f}'
                self.assertTrue(os.path.exists(check_path), msg=f'Could not find {check_path}')

    @pytest.mark.run(order=4)
    def test_validate(self):
        cmd = 'metamodel.py validate -f tests/integration/data/smoff.json -a smoff_test'
        os.system(cmd)

        # Pick some random files to spot check their existence
        files_to_check = [
            'fig_validation_HeatingElectricity_RF.png',
            'fig_validation_ts_Winter_HeatingElectricity_RF.png',
            'metrics.csv',
        ]

        for d in self.output_dirs:
            for f in files_to_check:
                check_path = f'{d}/ValidationData/images/{f}'
                self.assertTrue(os.path.exists(check_path), msg=f'Could not find {check_path}')

