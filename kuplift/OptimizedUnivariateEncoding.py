######################################################################################
# Copyright (c) 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "kuplift - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################

"""Optimized Univariate Encoding

This module contains everything needed to make univariate variable transformation
optimized through the use of the C++ implementation of 'umodl'. It calls the
'umodl' executable as a subprocess indirectly by the use the 'umodl' library.

The main class of this module is 'OptimizedUnivariateEncoding'.

An example code is in examples/optimized_univariate_encoding.py.
"""

import pathlib
import tempfile
from warnings import warn
import math
import json
from typing import Optional, Sequence, Union
from itertools import starmap
from dataclasses import dataclass
from abc import ABC, abstractmethod
import khiops.sklearn.dataset
import pandas as pd
from umodl import run_umodl


class Partition(ABC):
    @property
    @abstractmethod
    def parts(self):
        pass

    def __iter__(self):
        return iter(self.parts)
    

class ValGrp(list):
    def __str__(self):
        return "{%s}" % ", ".join(self)


class ValGrpPartition(Partition):
    """Partition of type 'value groups'.
    
    Attributes
    ----------
        groups: Sequence[ValGrp]
            The groups. Each group is an iterable of its values.
        defaultgroupindex: int
            The group index affected to transformed elements when they do not explicitly appear in any group.
    """

    def __init__(self, groups: Sequence[ValGrp], defaultgroupindex: int):
        if not groups:
            raise ValueError("there must be at least one group")
        if defaultgroupindex < 0 or defaultgroupindex >= len(groups):
            raise ValueError(f"default group index is {defaultgroupindex} but groups are numbered from 0 to {len(groups) - 1}")
        self.groups = groups
        self.defaultgroupindex = defaultgroupindex

    @property
    def parts(self):
        return self.groups

    def transform(self, col):
        return col.transform(self.transform_elem)

    def transform_elem(self, elem):
        for i, group in enumerate(self.groups):
            if str(elem) in group:
                return i
        return self.defaultgroupindex
    
    def __repr__(self):
        return f"ValGrpPartition({self.groups!r}, {self.defaultgroupindex!r})"

    def __str__(self):
        return """
Value group partition
    {ngroups} groups ("*" indicates the default group):
{groups}
"""[1:-1].format(
    ngroups=len(self.groups),
    groups="\n".join(f"      {'*' if i == self.defaultgroupindex else ' '} - {group}" for i, group in enumerate(self.groups))
)


@dataclass
class Interval:
    lower: Optional[float] = None
    upper: Optional[float] = None
        
    @property
    def catches_missing(self):
        return self.lower is None or self.upper is None
    
    def __contains__(self, x):
        return not self.catches_missing and (self.lower <= x < self.upper)
    
    def __str__(self):
        return "[]" if self.catches_missing else f"[{self.lower}, {self.upper}["
    
    def __bool__(self):
        return not self.catches_missing


class IntervalPartition(Partition):
    """Partition of type 'intervals'.
    
    Attributes
    ----------
        intervals: Sequence[Interval]
            The intervals. Each interval is a pair defining its lower bound and its upper bound (in that order).
            The exception to this rule is the empty interval representing 'MISSING' values. If present, it must be
            the first interval of the sequence.
    """

    def __init__(self, intervals: Sequence[Interval]):
        if not intervals:
            raise ValueError("there must be at least one interval")
        if intervals[0].catches_missing:
            if len(intervals) >= 1:
                intervals[1].lower = -math.inf
                intervals[-1].upper = math.inf
        else:
            intervals[0].lower = -math.inf
            intervals[-1].upper = math.inf
        self.intervals = intervals

    @property
    def parts(self):
        return self.intervals

    def transform(self, col):
        return col.transform(self.transform_elem)

    def transform_elem(self, elem):
        if not isinstance(elem, (int, float)) or math.isnan(elem):
            if self.intervals[0].catches_missing:
                return 0
            raise ValueError(f"cannot transform element of type {type(elem)} when there is no dedicated 'MISSING' interval")
        for i, interval in enumerate(self.intervals):
            if elem in interval:
                return i
            
    def __repr__(self):
        return f"IntervalPartition({self.intervals!r})"
    
    def __str__(self):
        return """
Interval partition
    {nintervals} intervals:
{intervals}
"""[1:-1].format(
    nintervals=len(self.intervals),
    intervals="\n".join(f"      - {interval}" for interval in self.intervals)
)
    

@dataclass
class TargetTreatmentPair:
    """Target-treatment pair.

    Used to identify both a target and a treatment.
    This class only exists for the purpose of formatting.
    """

    target: object
    treatment: object

    def __hash__(self):
        return hash((self.target, self.treatment))
    
    def __str__(self):
        return f"P({self.target}|{self.treatment})"


class OptimizedUnivariateEncoding:
    """
    The OptimizedUnivariateEncoding class makes use of the external umodl tool hosted at https://github.com/UData-Orange/umodl.

    Attributes
    ----------
    model: dict mapping str to ValGrpPartition or IntervalPartition
        The model generated by the 'umodl' executable.
        It describes the partitioning of values of informative variables into groups or intervals.
        It maps the informative variable names to value partitions.

    levels: list of (str, float) pairs
        (variable-name, variable-level) pairs in decreasing level order.

    variable_cols: DataFrame
        The data columns of all variables.
        This means all the data from the dataset but the treatment and target columns.

    treatment_col: Series
        The treatment column from the dataset.

    target_col: Series
        The target column from the dataset.

    input_variables: list of str
        The names of the variables.

    treatment_name: str
        The name of the treatment column.

    target_name: str
        The name of the target column.

    treatment_modalities: list
        All the different treatments from the dataset.

    target_modalities: list
        All the different targets from the dataset.

    target_treatment_pairs: list of TargetTreatmentPair
        All (target, treatment) pairs.
    """

    def __init__(self):
        self.model: dict[str, Union[ValGrpPartition, IntervalPartition]] = {}
        self.levels: list[tuple[str, float]] = []
        self.variable_cols = None
        self.treatment_col = None
        self.target_col = None

    @property
    def input_variables(self):
        return self.variable_cols.columns.to_list()
    
    @property
    def treatment_name(self):
        return self.treatment_col.name
    
    @property
    def target_name(self):
        return self.target_col.name
    
    @property
    def treatment_modalities(self):
        return list(self.treatment_col.unique())
    
    @property
    def target_modalities(self):
        return list(self.target_col.unique())
    
    @property
    def target_treatment_pairs(self):
        return [TargetTreatmentPair(target, treatment) for target in self.target_modalities for treatment in self.treatment_modalities]

    def fit_transform(self, data, treatment_col, target_col, maxpartnumber = None):
        """Learn a discretisation model using UMODL and transform the data.

        Parameters
        ----------
        data: pd.DataFrame
            Dataframe containing feature variables. Categorical
            variables should have the object dtype, otherwise they
            are processed as numerical variables.
        treatment_col: pd.Series
            Treatment column.
        target_col: pd.Series
            Outcome column.
        maxpartnumber: int, default=None
            The maximal number of intervals or groups. None means default to the 'umodl' program default.

        Returns
        -------
        pd.Dataframe
            Pandas Dataframe that contains encoded data.
        """
        self.fit(data, treatment_col, target_col, maxpartnumber)
        return self.transform(data)
    
    def fit(self, data, treatment_col, target_col, maxpartnumber = None):
        """Learn a discretisation model using UMODL.

        Parameters
        ----------
        data: pd.DataFrame
            Dataframe containing feature variables. Categorical
            variables should have the object dtype, otherwise they
            are processed as numerical variables.
        treatment_col: pd.Series
            Treatment column.
        target_col: pd.Series
            Outcome column.
        maxpartnumber: int, default=None
            The maximal number of intervals or groups. None means default to the 'umodl' program default.
        """
        # Force the types of the treatment and target columns so that khiops lib understands they are categorical
        if treatment_col.dtype != object:
            warn("Treatment column's dtype is not object; fixing...")
            treatment_col = treatment_col.astype(object)
        if target_col.dtype != object:
            warn("Target column's dtype is not object; fixing...")
            target_col = target_col.astype(object)

        # Create temporary .txt (data) and .kdic (Khiops dictionary) files for use with the umodl executable
        with tempfile.TemporaryDirectory() as dirname:
            dirpath = pathlib.Path(dirname)
            kdicfilename = str(dirpath / "main_table.kdic")
            dataset = khiops.sklearn.dataset.Dataset(data.join([treatment_col, target_col]))

            txtfilename, _ = dataset.create_table_files_for_khiops(dirname)  # Create .txt file
            dataset.create_khiops_dictionary_domain().export_khiops_dictionary_file(kdicfilename)  # Create .kdic file
            run_umodl(txtfilename, kdicfilename, "main_table", treatment_col.name, target_col.name, maxpartnumber)

            txtfilepath = pathlib.Path(txtfilename)
            with open(txtfilepath.with_name(f"UP_{txtfilepath.stem}.json")) as jsonfile:
                docroot = json.load(jsonfile)

        model = {}
        for variable in docroot['detailed statistics']:
            vardim = variable['dataGrid']['dimensions'][0]
            if vardim['partitionType'] == 'Value groups':
                model[vardim['variable']] = ValGrpPartition(list(map(ValGrp, vardim['partition'])), vardim['defaultGroupIndex'])
            elif vardim['partitionType'] == 'Intervals':
                model[vardim['variable']] = IntervalPartition(list(starmap(Interval, vardim['partition'])))
            else: raise ValueError("unsupported partition type")

        levels = sorted(
            ((attr['name'], attr['level']) for attr in docroot['attributes']),
            key=lambda namelevel: (-namelevel[1], namelevel[0])
        )

        # Only write to the instance's attributes if all the above succeeded.
        self.model = model
        self.levels = levels
        self.variable_cols = data
        self.treatment_col = treatment_col
        self.target_col = target_col


    def transform(self, data):
        """Apply the discretisation model learned by the fit() method.

        Parameters
        ----------
        data: pd.DataFrame
            Dataframe containing feature variables.

        Returns
        -------
        pd.DataFrame
            Pandas Dataframe that contains encoded data.
        """
        data = data[list(self.model.keys())]  # Keep only informative variables
        self.transformed_data = data.transform(lambda col: self.model[col.name].transform(col))
        return self.transformed_data
    
    def get_levels(self):
        """Get the level of all variables.
        
        Returns
        -------
        list[tuple[str, float]]
            (variable-name, variable-level) pairs in decreasing level order.
        """
        return self.levels
    
    def get_partition(self, variable):
        """Get the partition corresponding to a single variable of the model.

        Parameters
        ----------
        variable: str
            The variable name.
        
        Returns
        -------
        ValGrpPartition | IntervalPartition
            The partition corresponding to a single variable of the model.
        """
        return self.model[variable]
    
    def get_target_frequencies(self, variable):
        """Get the frequencies for each (target, treatment) pair.
        
        The frequencies are computed for a single variable.
        
        Parameters
        ----------
        variable: str
            The variable name.

        Returns
        -------
        pd.DataFrame
            The frequencies as a Dataframe containing:
                - A column named 'Part' listing all the parts of the variable.
                - One column per (target, treatment) pair.
        """
        varcol = self.variable_cols[variable]
        partition = self.get_partition(variable)
        return pd.DataFrame(
            {
                **{"Part": partition},
                **{
                    ttpair: [
                        len(
                            varcol[
                                (self.treatment_col == ttpair.treatment) & (self.target_col == ttpair.target) & varcol.map(lambda elem: partition.transform_elem(elem) == i)
                            ]
                        )
                        for i, _ in enumerate(partition)
                    ]
                    for ttpair in self.target_treatment_pairs
                }
            }
        )
    
    def get_target_probabilities(self, variable):
        """Get the probabilities P(target|treatment) for each (target, treatment) pair.
        
        The probabilities are computed for a single variable.
        
        Parameters
        ----------
        variable: str
            The variable name.

        Returns
        -------
        pd.DataFrame
            The probabilities as a Dataframe containing:
                - A column named 'Part' listing all the parts of the variable.
                - One column per (target, treatment) pair.
        """
        return self.get_target_frequencies(variable).transform({
            "Part": lambda x: x,
            **{ttpair: lambda x: x / len(self.variable_cols) for ttpair in self.target_treatment_pairs}
        })
    
    def get_uplift(self, reftarget, reftreatment, variable):
        """Get the uplift for a single variable.

        See explanations of the computations in the 'Returns' section below.
        
        Parameters
        ----------
        reftarget
            The reference target.
        reftreatment
            The reference treatment to which all the other treatments are compared.
        variable: str
            The name of the variable.

        Returns
        -------
        pd.DataFrame
            A Dataframe containing:
                - A column named 'Part' listing all the parts of the variable.
                - One column per treatment other than the reference treatment.
                  A column gives the difference P(reftarget|treatment) - P(reftarget|reftreatment), that is,
                  the benefit (or deficit) of probabilities to have 'reftarget' as the outcome with the column's
                  treatment compared to the reference treatment.
        """
        probs = self.get_target_probabilities(variable)
        refprobs = probs[TargetTreatmentPair(reftarget, reftreatment)]
        return probs["Part"].to_frame().join(pd.DataFrame({
            f"Uplift {reftarget} {treatment}": probs[TargetTreatmentPair(reftarget, treatment)] - refprobs
            for treatment in self.treatment_modalities if treatment != reftreatment
        }))
