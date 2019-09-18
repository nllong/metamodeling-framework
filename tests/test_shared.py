# -*- coding: utf-8 -*-
"""
.. moduleauthor:: Nicholas Long (nicholas.l.long@colorado.edu, nicholas.lee.long@gmail.com)
"""

from unittest import TestCase

from metamodeling.shared import is_int


class TestShared(TestCase):
    def test_is_int(self):
        self.assertTrue(is_int(5.6))
        self.assertFalse(is_int("Not an int"))
