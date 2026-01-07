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
import math
import json
from typing import Optional, Sequence, Union
from itertools import starmap
from dataclasses import dataclass
from textwrap import indent
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
        groups : Sequence[ValGrp]
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
        def formatgroupline(group, isdefault):
            return f"  {'*' if isdefault else ' '} - {group}"
        return """
Value group partition
    {ngroups} groups ("*" indicates the default group):
{groups}
"""[1:-1].format(
    ngroups=len(self.groups),
    groups=indent("\n".join(formatgroupline(group, i == self.defaultgroupindex) for i, group in enumerate(self.groups)), 4 * " ")
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
        def formatintervalline(interval):
            return f"  - {interval}"
        return """
Interval partition
    {nintervals} intervals:
{intervals}
"""[1:-1].format(
    nintervals=len(self.intervals),
    intervals=indent("\n".join(map(formatintervalline, self.intervals)), 4 * " ")
)
    

@dataclass
class ProbSpec:
    target: object
    treatment: object

    def __hash__(self):
        return hash((self.target, self.treatment))
    
    def __str__(self):
        return f"P({self.target}|{self.treatment})"


class OptimizedUnivariateEncoding:
    """
    The OptimizedUnivariateEncoding class makes use of the external umodl tool hosted at https://github.com/UData-Orange/umodl
    """

    def __init__(self):
        self.model: dict[str, Union[ValGrpPartition, IntervalPartition]] = {}
        self.target_probs: dict[str, pd.DataFrame] = {}
        self.uplift: dict[str, pd.DataFrame] = {}

    def fit_transform(self, data, treatment_col, target_col, maxpartnumber = None):
        """fit_transform() learns a discretisation model using UMODL and transforms the data.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe containing feature variables. Categorical
            variables should have the object dtype, otherwise they
            are processed as numerical variables.
        treatment_col : pd.Series
            Treatment column.
        target_col : pd.Series
            Outcome column.
        maxpartnumber : int, default=None
            The maximal number of intervals or groups. None means default to the 'umodl' program default.

        Returns
        -------
        pd.Dataframe
            Pandas Dataframe that contains encoded data.
        """
        self.fit(data, treatment_col, target_col, maxpartnumber)
        return self.transform(data)
    
    def fit(self, data, treatment_col, target_col, maxpartnumber = None):
        """fit() learns a discretisation model using UMODL.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe containing feature variables. Categorical
            variables should have the object dtype, otherwise they
            are processed as numerical variables.
        treatment_col : pd.Series
            Treatment column.
        target_col : pd.Series
            Outcome column.
        maxpartnumber : int, default=None
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

            self.model = {}
            for variable in docroot['detailed statistics']:
                vardim = variable['dataGrid']['dimensions'][0]
                if vardim['partitionType'] == 'Value groups':
                    self.model[vardim['variable']] = ValGrpPartition(list(map(ValGrp, vardim['partition'])), vardim['defaultGroupIndex'])
                elif vardim['partitionType'] == 'Intervals':
                    self.model[vardim['variable']] = IntervalPartition(list(starmap(Interval, vardim['partition'])))
                else: raise ValueError("unsupported partition type")

        self.variable_cols = data
        self.treatment_col = treatment_col
        self.target_col = target_col
        self.treatments = self.treatment_col.unique()
        self.targets = self.target_col.unique()

    def transform(self, data):
        """transform() applies the discretisation model learned by the fit() method.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe containing feature variables.

        Returns
        -------
        pd.DataFrame
            Pandas Dataframe that contains encoded data.
        """
        data = data[list(self.model.keys())]  # Keep only informative variables
        self.transformed_data = data.transform(lambda col: self.model[col.name].transform(col))
        return self.transformed_data
    
    def get_partition(self, variable):
        """get_partition() gets the partition corresponding to a single variable of the model.

        Parameters
        ----------
        variable : str
            The variable name.
        
        Returns
        -------
        ValGrpPartition | IntervalPartition
            The partition corresponding to a single variable of the model.
        """
        return self.model[variable]
    
    def get_target_probability(self, variable):
        """get_target_probability() gets the probabilities P(target|treatment) for each (target, treatment) pair.
        
        The probabilities are computed for a single variable.
        The results are both stored in the 'self.target_probs' dictionary for future reference and returned for
        convenience.

        When called with the same variable as a previous call, in will not perform any calculation and will simply
        return the entry already stored in 'self.target_probs'.
        
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
        if variable in self.target_probs:
            return self.target_probs[variable]
        varcol = self.variable_cols[variable]
        treatment_target_pairs = [ProbSpec(target, treatment) for treatment in self.treatments for target in self.targets]
        partition = self.get_partition(variable)
        self.target_probs[variable] = pd.DataFrame(
            {
                **{"Part": partition},
                **{
                    probspec: [
                        len(
                            varcol[
                                (self.treatment_col == probspec.treatment) & (self.target_col == probspec.target) & varcol.map(lambda elem: partition.transform_elem(elem) == i)
                            ]
                        ) / len(varcol)
                        for i, _ in enumerate(partition)
                    ]
                    for probspec in treatment_target_pairs
                }
            }
        )
        return self.target_probs[variable]
    
    def get_uplift(self, reftarget, reftreatment, variable):
        """get_uplift() gets the uplift for a single variable.

        The results are both stored in the 'self.uplift' dictionary for future reference and returned for
        convenience.

        The probabilities used for computations are the ones stored in the 'self.target_probs' dictionary.
        These should have been previously populated by a call to 'get_target_probability' with the same variable
        as specified in the call to this function.
        See explanations of the computations in the 'Returns' section below.

        When called with the same variable as a previous call, in will not perform any calculation and will simply
        return the entry already stored in 'self.uplift'.
        
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
                - A column named 'Part' as for the target probability Dataframes stored in 'self.target_probs'.
                - One column per treatment other than the reference treatment.
                  A column gives the difference P(reftarget|treatment) - P(reftarget|reftreatment), that is,
                  the benefit (or deficit) of probabilities to have 'reftarget' as the outcome with the column's
                  treatment compared to the reference treatment.
        """
        if variable in self.uplift:
            return self.uplift[variable]
        # 'tut(s)': Treatment(s) Under Test
        tuts = [t for t in self.treatments if t != reftreatment]
        refprobs = self.target_probs[variable][ProbSpec(reftarget, reftreatment)]
        self.uplift[variable] = pd.DataFrame(
            {
                **{"Part": self.target_probs[variable]["Part"]},
                **{
                    f"Up {reftarget} {treatment}": self.target_probs[variable][ProbSpec(reftarget, treatment)] - refprobs
                    for treatment in tuts
                }
            }
        )
        return self.uplift[variable]
