######################################################################################
# Copyright (c) 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "kuplift - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
import pathlib
import tempfile
from warnings import warn
import json
import khiops.sklearn.dataset
from umodl import run_umodl


class ValGrpPartition:
    """Partition of type 'value groups'."""
    def __init__(self, groups, defaultgroupindex):
        self.groups = groups
        self.defaultgroupindex = defaultgroupindex

    def transform(self, col):
        return col.transform(self._transform_elem)

    def _transform_elem(self, elem):
        for i, group in enumerate(self.groups):
            if str(elem) in group:
                return i
        return self.defaultgroupindex


class IntervalPartition:
    """Partition of type 'intervals'."""
    def __init__(self, intervals):
        self.intervals = intervals
        first_non_missing_interval_index = next(i for i, interval in enumerate(self.intervals) if interval)
        self.intervals[first_non_missing_interval_index] = (float('-inf'), self.intervals[first_non_missing_interval_index][1])
        self.intervals[-1] = (self.intervals[-1][0], float('inf'))

    def transform(self, col):
        return col.transform(self._transform_elem)

    def _transform_elem(self, elem):
        for i, interval in enumerate(self.intervals):
            if interval and interval[0] <= elem < interval[1]:
                return i
        return self.intervals.index(())


class OptimizedUnivariateEncoding:
    """
    The OptimizedUnivariateEncoding class makes use of the external umodl tool hosted at https://github.com/UData-Orange/umodl
    """

    def __init__(self):
        self._model = None

    def fit_transform(self, data, treatment_col, y_col, maxpartnumber = None):
        """
        fit_transform() learns a discretisation model using UMODL and
        transforms the data.

        Parameters
        ----------
        data : pd.Dataframe
            Dataframe containing feature variables. Categorical
            variables should have the object dtype, otherwise they
            are processed as numerical variables.
        treatment_col : pd.Series
            Treatment column.
        y_col : pd.Series
            Outcome column.
        maxpartnumber : int, default=None
            The maximal number of intervals or groups. None means default to the 'umodl' program default.

        Returns
        -------
        pd.Dataframe
            Pandas Dataframe that contains encoded data.
        """
        self.fit(data, treatment_col, y_col, maxpartnumber)
        return self.transform(data)
    
    def fit(self, data, treatment_col, y_col, maxpartnumber = None):
        """
        fit() learns a discretisation model using UMODL.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe containing feature variables. Categorical
            variables should have the object dtype, otherwise they
            are processed as numerical variables.
        treatment_col : pd.Series
            Treatment column.
        y_col : pd.Series
            Outcome column.
        maxpartnumber : int, default=None
            The maximal number of intervals or groups. None means default to the 'umodl' program default.
        """
        # Force the types of the treatment and target columns so that khiops lib understands they are categorical
        if treatment_col.dtype != object:
            warn("Treatment column's dtype is not object; fixing...")
            treatment_col = treatment_col.astype(object)
        if y_col.dtype != object:
            warn("Target column's dtype is not object; fixing...")
            y_col = y_col.astype(object)

        # Create temporary .txt (data) and .kdic (Khiops dictionary) files for use with the umodl executable
        with tempfile.TemporaryDirectory() as dirname:
            dirpath = pathlib.Path(dirname)
            kdicfilename = str(dirpath / "main_table.kdic")
            dataset = khiops.sklearn.dataset.Dataset(data.join([treatment_col, y_col]))

            txtfilename, _ = dataset.create_table_files_for_khiops(dirname)  # Create .txt file
            dataset.create_khiops_dictionary_domain().export_khiops_dictionary_file(kdicfilename)  # Create .kdic file
            run_umodl(txtfilename, kdicfilename, "main_table", treatment_col.name, y_col.name, maxpartnumber)

            txtfilepath = pathlib.Path(txtfilename)
            with open(txtfilepath.with_name(f"UP_{txtfilepath.stem}.json")) as jsonfile:
                docroot = json.load(jsonfile)

            self._model = {}
            for variable in docroot['detailed statistics']:
                vardim = variable['dataGrid']['dimensions'][0]
                if vardim['partitionType'] == 'Value groups':
                    self._model[vardim['variable']] = ValGrpPartition(vardim['partition'], vardim['defaultGroupIndex'])
                elif vardim['partitionType'] == 'Intervals':
                    self._model[vardim['variable']] = IntervalPartition(list(map(tuple, vardim['partition'])))
                else: raise ValueError("unsupported partition type")

    def transform(self, data):
        """
        transform() applies the discretisation model learned by the
        fit() method.

        Parameters
        ----------
        data : pd.Dataframe
            Dataframe containing feature variables.

        Returns
        -------
        pd.Dataframe
            Pandas Dataframe that contains encoded data.
        """
        data = data[list(self._model.keys())]  # Keep only informative variables
        return data.transform(lambda col: self._model[col.name].transform(col))
