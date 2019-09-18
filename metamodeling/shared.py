# -*- coding: utf-8 -*-
"""
.. moduleauthor:: Nicholas Long (nicholas.l.long@colorado.edu, nicholas.lee.long@gmail.com)
"""
# import cPickle  # python 2
import pickle
import csv
import gzip
import math
import os
import re


def apply_cyclic_transform(row, column_name, category_count):
    return math.sin(2 * math.pi * row[column_name] / category_count)


def pickle_file(obj, filename, gzipfile=False):
    """

    :param obj:
    :param filename: Filename, without the extension
    :param gzipfile:
    """
    if gzipfile:
        gfile = gzip.open('%s.pklzip' % filename, 'wb')
    else:
        gfile = open('%s.pkl' % filename, 'wb')
    pickle.dump(obj, gfile)
    gfile.close()


def convert(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def unpickle_file(filename):
    extension = os.path.splitext(filename)[1]
    if extension == '.pklzip':
        gfile = gzip.open(filename, 'rb')
    else:
        gfile = open(filename, 'rb')
    return pickle.load(gfile)


def save_dict_to_csv(data, filename):
    with open(filename, 'w') as cfile:
        writer = csv.DictWriter(cfile, data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def zipdir(path, ziph, extension=None):
    """
    Flattened zip directory
    :param path:
    :param ziph:
    :param extension:
    """
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for a_file in files:
            filename = os.path.join(root, a_file)
            if extension:
                if a_file.endswith(extension):
                    ziph.write(filename, os.path.basename(filename))
            else:
                ziph.write(filename, os.path.basename(filename))


def is_int(value):
    try:
        int(value)
    except ValueError:
        return False
    return True
