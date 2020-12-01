"""
.. moduleauthor:: Nicholas Long (nicholas.l.long@colorado.edu, nicholas.lee.long@gmail.com)
"""
import copy
import json
import os


class JsonProcessorFile(object):
    """Generate a dict of processing options that exist in a dictionary of dictionaries. Allow renaming
    of the fields. The results of this class is used to flatten out a JSON into CSV style.

    For example the following Dict below will generate another dictionary outlined below.

    Limitations: only works with 2 levels of dictionaries

    .. code-block:: python

        {
            "internal_loads_multiplier": {
                "lpd_multiplier": 0.7544625053841931,
                "epd_multiplier": 1.0,
                "people_per_floor_area_multiplier": 0.8572429796331562,
                "lpd_average": 7.30887013864965,
                "epd_average": 8.07293281253229,
                "ppl_average": 0.046136433190623,
                "applicable": true
            },
        }

    .. code-block:: python

        {
            level_1: 'internal_loads_multiplier',
            level_2: 'lpd_multiplier',
            rename_to: '',
            order: 1
        },
        {
            level_1: 'internal_loads_multiplier',
            level_2: 'epd_multiplier',
            rename_to: '',
            order: 1
        },
        {
            level_1: 'internal_loads_multiplier',
            level_2: 'lpd_average',
            rename_to: '',
            order: 1
        },
    """

    def __init__(self, json_files):
        """
        :param json_files: list of files to process
        """
        self.files = json_files
        self.data = []

        self.process()

    def process(self):
        """Process the list of json files"""
        for file in self.files:
            data = {
                "file": os.path.basename(file),
                "data": []
            }

            with open(file) as f:
                f = json.load(f)
                for k, v in f.items():
                    new_var = {
                        "level_1": k,
                        "level_2": None,
                        "rename_to": "",  # if there is no rename_to, then the name is set to the key
                        "order": 1,  # if there are duplicates, then the fields will be sorted alphabetically
                    }
                    if isinstance(v, dict):
                        # The value is a dict, so process the dict values too
                        for k2, v2 in v.items():
                            new_var_2 = copy.deepcopy(new_var)
                            new_var_2["level_2"] = k2

                            data["data"].append(new_var_2)
                    else:
                        # single key -- just save the new variable
                        data["data"].append(new_var)

            self.data.append(data)

    def save_as(self, filename):
        """Save the format to be used in the post_processor scripts"""
        if os.path.exists(filename):
            print(f"File already exists, will not overwrite, {filename}")
            return False
        else:
            with open(filename, 'w') as f:
                json.dump(self.data, f, indent=2)
            return True
