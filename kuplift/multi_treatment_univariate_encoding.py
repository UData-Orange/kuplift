# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

"""Multi-treatment Univariate Encoding

This module contains everything needed to make univariate variable transformation
capable of merging treatments that give similar outcome.

The main class of this module is 'MultiTreatmentUnivariateEncoding'.
"""

from pathlib import Path
from .helperfunctions import partition_to_rule, random_varname
from tempfile import TemporaryDirectory
import logging
from itertools import chain
from functools import cached_property
from dataclasses import dataclass
import collections.abc
import typing
import warnings
from operator import itemgetter
import khiops.core
import numpy
import pandas
import pandas.core.dtypes.base

logger = logging.getLogger(__name__)


VarType = typing.Literal["Numerical", "Categorical"]
Part = khiops.core.PartInterval | khiops.core.PartValueGroup
Groups = tuple[tuple[str]]


@dataclass(frozen=True)
class DatasetInfo:
    """Basic information of a dataset.
    
    Attributes
    ----------
    jname: str
        The name of the target/outcome variable.
    tname: str
        The name of the treatment variable.
    xs: dict mapping str to VarType
        Dictionary mapping variable names to variable types.
    size: int
        The number of observations.
    """
    jname: str
    tname: str
    xs: dict[str, VarType]
    size: int

    @property
    def categorical_xs(self) -> list[str]:
        return [varname for varname, vartype in self.xs.items() if vartype == "Categorical"]


@dataclass(frozen=True)
class VarStats:
    """Statistics for a variable, extracted from a Khiops analysis report.
    
    Attributes
    ----------
    is_informative: bool
        `true` if the variable is informative, `false` otherwise.

    level: float
        The level. Zero for a non-informative variable.

    parts: list[Part]
        The list of parts. Empty for a non-informative variable.

    nijt: pandas DataFrame or None
        `None` for a non-informative variable.
        Otherwise, a DataFrame that is the N_ijt table of the variable, where:
            N: number of observations;
            i: part (interval for a numerical variable or value group for a categorical variable);
            j: target (outcome);
            t: treatment.
        One DataFrame column contains the number of observations for one part (one "i").
        One DataFrame row contains the numbers of observations for one target-treatment pair (one "(j, t)" pair).
    """
    is_informative: bool
    level: float
    parts: list[Part]
    nijt: pandas.DataFrame | None


@dataclass(frozen=True)
class VarStatsWithGroups:
    varstats: VarStats
    groups: dict[Part, Groups]


@dataclass(frozen=True)
class GeneralStats:
    """Statistics extracted from a Khiops analysis report.
    
    Attributes
    ----------
    js: list
        The list of targets.

    ts: list of str
        The list of treatments.

    jts: list of str
        The list of target-treatment pairs.

    informative_xnames: list
        The list of informative input variables.

    noninformative_xnames: list
        The list of non-informative input variables.
    """
    js: list
    ts: list
    jts: list
    informative_xnames: list[str]
    noninformative_xnames: list[str]


@dataclass(frozen=True)
class Stats:
    """Statistics extracted from a Khiops analysis report.
    
    Attributes
    ----------
    generalstats: GeneralStats
        Statistics not including input-variable specific stats.

    xstats: dict mapping str to VarStats
        A dictionary mapping the names of the input variables to the statistics of these variables.
    """
    generalstats: GeneralStats
    xstats: dict[str, VarStats]


@dataclass(frozen=True)
class StatsWithGroups:
    generalstats: GeneralStats
    xstats: dict[str, VarStatsWithGroups]


@dataclass(frozen=True)
class UpliftDictionary:
    """Khiops dictionary with extra information.
    
    Attributes
    ----------
    domain: khiops.core.DictionaryDomain
        The domain to which the dictionary belongs.
    dict: khiops.core.Dictionary
        The Khiops dictionary.
    jtname: str
        The name of the created target|treatment calculated variable.
    """
    domain: khiops.core.DictionaryDomain
    dict: khiops.core.Dictionary
    jtname: str


@dataclass(frozen=True)
class FileOutput:
    outputdir: Path
    is_persistent: bool
    @cached_property
    def datasetfile(self) -> Path:
        return self.outputdir / "dataset.csv"
    @cached_property
    def dictfile(self) -> Path:
        return self.outputdir / "dictionary.kdic"
    @cached_property
    def analysisresultdir(self) -> Path:
        return self.outputdir / "analysis_result"
    @cached_property
    def predictor_analysisresultfile(self) -> Path:
        return self.analysisresultdir / "predictor_analysis_result.khj"
    def xi_analysisresultfile(self, xname: str, iname: str) -> Path:
        return self.analysisresultdir / "{xname}_{iname}_analysis_result.khj".format(xname=xname, iname=iname)
    @cached_property
    def is_temporary(self) -> bool:
        return not self.is_persistent
    def __str__(self) -> str:
        xname = "x_example"
        iname = "I_example"
        fileoutputtype = "Persistent" if self.is_persistent else "Temporary"
        return "{type} file output: outputdir={outdir}, datasetfile={dsf}, dictfile={dctf}, "\
            "analysisresultdir={ard}, predictor_analysisresultfile={parf}, xi_analysisresultfile({x!r}, {i!r})={xiarf}".format(
                type=fileoutputtype,
                outdir=self.outputdir,
                dsf=self.datasetfile,
                dctf=self.dictfile,
                ard=self.analysisresultdir,
                parf=self.predictor_analysisresultfile,
                x=xname,
                i=iname,
                xiarf=self.xi_analysisresultfile(xname, iname)
            )


def build_ijt_table_by_column(xname: str, jts: list[str], parts: list[Part], func: collections.abc.Callable[[int, Part], typing.Sequence]) -> pandas.DataFrame:
    """Build a "*something*_ijt" table.

    Build a "*something*_ijt" table where *ijt* stands for:
    - *i*: part (interval for a numerical variable or value group for a categorical variable);
    - *j*: target (outcome);
    - *t*: treatment.
    One DataFrame column contains the values for one part (one part = one "i").
    One DataFrame row contains the values for one target-treatment pair (one "(j, t)" pair).
    """
    return pandas.DataFrame(index=jts, data={i: func(iindex, i) for iindex, i in enumerate(parts)})


def build_ijt_table_by_cell(xname: str, jts: list[str], parts: list[Part], func: collections.abc.Callable[[int, Part, int, str], typing.Any]) -> pandas.DataFrame:
    """Build a "<something>_ijt" table.

    Build a "<something>_ijt" table where *ijt* stands for:
    - *i*: part (interval for a numerical variable or value group for a categorical variable);
    - *j*: target (outcome);
    - *t*: treatment.
    One DataFrame column contains the values for one part (one part = one "i").
    One DataFrame row contains the values for one target-treatment pair (one "(j, t)" pair).
    """
    return pandas.DataFrame(index=jts, data={i: [func(iindex, i, jtindex, jt) for jtindex, jt in enumerate(jts)] for iindex, i in enumerate(parts)})


class MultiTreatmentUnivariateEncoding:
    """
    The MultiTreatmentUnivariateEncoding class makes use of the khiops Python wrapper
    and enables one to fit and transform data while grouping treatments giving similar
    outcome.
    
    Attributes
    ----------
    variable_cols: DataFrame
        The data columns of all variables.
        This means all the data from the dataset but the treatment and target columns.

    treatment_col: Series
        The treatment column from the dataset.

    target_col: Series
        The target column from the dataset.

    stats: StatsWithGroups
        Statistics produced by Khiops, augmented with treatment groups, also computed by Khiops.
    """

    def __init__(self):
        self.variable_cols = None
        self.treatment_col = None
        self.target_col = None
        self.stats = None


    @property
    def input_variables(self) -> list[str]:
        """The names of the variables."""
        return list(self.stats.xstats)
    

    @property
    def informative_input_variables(self) -> list[str]:
        """The names of the informative variables."""
        return self.stats.generalstats.informative_xnames
    

    @property
    def noninformative_input_variables(self) -> list[str]:
        """The names of the non-informative variables."""
        return self.stats.generalstats.noninformative_xnames
    

    @property
    def treatment_name(self) -> str:
        """The name of the treatment column."""
        return self.treatment_col.name
    

    @property
    def target_name(self) -> str:
        """The name of the target column."""
        return self.target_col.name
    

    @property
    def treatment_modalities(self) -> list:
        """All the different treatments from the dataset."""
        return self.stats.generalstats.ts
    

    @property
    def target_modalities(self) -> list:
        """All the different targets from the dataset."""
        return self.stats.generalstats.js
    

    @property
    def target_treatment_pairs(self) -> list[str]:
        """All (target, treatment) pairs as a list of "target|treatment"-formatted strings."""
        return self.stats.generalstats.jts
    

    def fit(self, data, treatment_col, target_col, maxparts = None, maxtreatmentgroups = None, *, outputdir = None) -> None:
        """Learn a discretisation model using Khiops.

        Parameters
        ----------
        data: pandas.DataFrame
            Dataframe containing feature variables. Categorical
            variables should have the object dtype, otherwise they
            are processed as numerical variables.
        treatment_col: pandas.Series
            Treatment column.
        target_col: pandas.Series
            Outcome column.
        maxpartnumber: int, default=None
            The maximal number of intervals or groups. None means default to the 'khiops' program default.
        maxtreatmentgroups: int, default=None
            The maximal number of groups to define when grouping treatments together. None means automatic.
        outputdir: Path-like
            Set this if you want khiops's workfiles to be kept in a specific directory. If None, fallback
            to the default behaviour which is to have khiops write its files into a temporary directory that
            is deleted when the work is done.
        """
        is_outputdir_persistent = outputdir is not None
        if is_outputdir_persistent:
            dirpath = Path(outputdir)
        else:
            tmpdir = TemporaryDirectory()
            dirpath = Path(tmpdir.name)
        fileoutput = FileOutput(dirpath, is_outputdir_persistent)
        logger.info("%s", fileoutput)
        
        dataset = pandas.DataFrame(data).join([treatment_col, target_col])
        datasetinfo = DatasetInfo(target_col.name, treatment_col.name, vartypes_by_name_from_dataframe(data), len(dataset))

        stats, upliftdict = compute_stats(dataset, datasetinfo, fileoutput, maxparts)

        # Disable all input variables since we will create a selection variable for each one in turn
        # and it is that selection variable that will be enabled.
        for xname in stats.xstats:
            upliftdict.dict.get_variable(xname).used = False

        xstats_with_groups = {}
        for xname in stats.xstats:
            xstats_with_groups[xname] = group_treatments_for_variable(xname, datasetinfo, stats, upliftdict, fileoutput, maxtreatmentgroups)

        self.variable_cols = data
        self.treatment_col = treatment_col
        self.target_col = target_col
        self.stats = StatsWithGroups(stats.generalstats, xstats_with_groups)


    def transform(self, data):
        """Apply the discretisation model learned by the fit() method.

        Parameters
        ----------
        data: pandas.DataFrame
            Dataframe containing feature variables.

        Returns
        -------
        pandas.DataFrame
            Pandas Dataframe that contains encoded data.
        """
        data = data[list(self.model.keys())]  # Keep only informative variables
        self.transformed_data = data.transform(lambda col: self.model[col.name].transform(col))
        return self.transformed_data

    
    def fit_transform(self, data, treatment_col, target_col, *, maxpartnumber = None, maxtreatmentgroups = None, outputdir = None):
        """Learn a discretisation model using Khiops and transform the data.

        Parameters
        ----------
        data: pandas.DataFrame
            Dataframe containing feature variables. Categorical
            variables should have the object dtype, otherwise they
            are processed as numerical variables.
        treatment_col: pandas.Series
            Treatment column.
        target_col: pandas.Series
            Outcome column.
        maxpartnumber: int, default=None
            The maximal number of intervals or groups. None means default to the 'khiops' program default.
        maxtreatmentgroups: int, default=None
            The maximal number of groups to define when grouping treatments together. None means automatic.
        outputdir: Path-like
            Set this if you want khiops's workfiles to be kept in a specific directory. If None, fallback
            to the default behaviour which is to have khiops write its files into a temporary directory that
            is deleted when the work is done.

        Returns
        -------
        pandas.Dataframe
            Pandas Dataframe that contains encoded data.
        """
        self.fit(data, treatment_col, target_col, maxpartnumber=maxpartnumber, maxtreatmentgroups=maxtreatmentgroups, outputdir=outputdir)
        return self.transform(data)
    

    def get_levels(self):
        """Get the level of all variables.
        
        Returns
        -------
        list[tuple[str, float]]
            (variable-name, variable-level) pairs in decreasing level order.
        """
        return list(sorted(((varname, varstats.varstats.level) for varname, varstats in self.stats.xstats.items()), key=lambda varpair: (-varpair[1], varpair[0])))
    

    def get_level(self, variable):
        """Get the level of a single variable.

        Parameters
        ----------
        variable: str
            The variable to get the level from.
        
        Returns
        -------
        float
            The level of the specified variable.
        """
        return self.stats.xstats[variable].varstats.level
    

    def get_partition(self, variable: str) -> list[Part]:
        """Get the partition corresponding to a single variable of the model.

        Parameters
        ----------
        variable: str
            The variable name.
        
        Returns
        -------
        list[Part]
            The partition corresponding to a single variable of the model.
        """
        return self.stats.xstats[variable].varstats.parts
    

    def get_treatment_groups(self, variable: str | None = None) -> dict[Part, Groups] | dict[str, dict[Part, Groups]]:
        """Get the groups of treatments for one or all variables.

        Parameters
        ----------
        variable: str | None
            If set to None, get groups of all variables, otherwise get groups of specified variable.

        Returns
        -------
        If `variable` is not None, returns a dict mapping parts to treatment groups.
        If `variable` is None, returns a dict mapping variable names to dictionaries mapping parts to treatment groups.
        """
        if variable is not None:
            return self.stats.xstats[variable].groups
        else:
            return {xname: xstats.groups for xname, xstats in self.stats.xstats.items()}
    

    def get_target_frequencies(self, variable: str) -> pandas.DataFrame:
        """Get the frequencies N_ijt for a variable.
        
        Parameters
        ----------
        variable: str
            The variable name.

        Returns
        -------
        pandas.DataFrame
            The frequency table for the variable.
        """
        return self.stats.xstats[variable].varstats.nijt


    def get_target_frequencies_of_treatment_groups(self, variable):
        """Get the frequencies for each (target, treatment group) pair, for each part in the variable's partition.
        
        The frequencies are computed for a single variable.
        
        Parameters
        ----------
        variable: str
            The variable name.

        Returns
        -------
        dict[Part, pandas.Series]
            The frequencies as a dict mapping parts to Series which index represents target-treatmentgroup pairs and which values are the frequencies.
        """
        raise NotImplementedError
    

    def get_target_probabilities(self, variable):
        """Get the probabilities P(1)_ijt for a variable.
        
        Parameters
        ----------
        variable: str
            The variable name.

        Returns
        -------
        pandas.DataFrame
            The probability table for the variable.
        """
        raise NotImplementedError
    

    def get_target_probabilities_of_treatment_groups(self, variable):
        """Get the probabilities P(target|treatment group) for each (target, treatment group) pair.
        
        The probabilities are computed for a single variable.
        
        Parameters
        ----------
        variable: str
            The variable name.

        Returns
        -------
        dict[Part, pandas.Series]
            The probabilities as a dict mapping parts to Series which index represents target-treatmentgroup pairs and which values are the probabilities.
        """
        raise NotImplementedError
    

    def get_uplift(self, reftarget, reftreatment, variable):
        """Get the uplift Uplift_ijt for a variable.

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
        pandas.DataFrame
            A Dataframe containing:
                - A column named 'Part' listing all the parts of the variable.
                - One column per treatment other than the reference treatment.
                  A column gives the difference P(reftarget|treatment) - P(reftarget|reftreatment), that is,
                  the benefit (or deficit) of probabilities to have 'reftarget' as the outcome with the column's
                  treatment compared to the reference treatment.
        """
        raise NotImplementedError
    

    def get_uplift_of_treatment_groups(self, reftarget, reftreatment, variable):
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

        dict[Part, pandas.Series]
            The probabilities as a dict mapping parts to Series which index represents target-treatmentgroup pairs and
            which values are the differences P(reftarget|treatment group) - P(reftarget|reftreatment).
        """
        raise NotImplementedError
    

@dataclass(frozen=True)
class DimensionConstraint:
    type: str | None
    partition_type: str | None
    def validate(self, dimension: khiops.core.analysis_results.DataGridDimension):
        if self.type is not None:
            if dimension.type != self.type:
                raise RuntimeError("type {!r} of variable {!r} does not match expected type {!r}".format(
                    dimension.type, dimension.variable, self.type))
        if self.partition_type is not None:
            if dimension.partition_type != self.partition_type:
                raise RuntimeError("partition type {!r} of variable {!r} does not match expected partition type {!r}".format(
                    dimension.partition_type, dimension.variable, self.partition_type))


def find_dimensions(variables: typing.Collection[str] | typing.Mapping[str, DimensionConstraint], dimensions: typing.Iterable[khiops.core.analysis_results.DataGridDimension]):
    """Find specific dimensions in a dimension list.

    Parameters
    ----------
    variables: mapping or non-mapping collection
        If it is a non-mapping collection, it contains the names of the variables.
        If it is a mapping collection, it maps the names of the variables to dimension constraints.
        The dimension contraint may be `None` to indicate that there is no constraint attached to a variable.

    dimensions: iterable of `khiops.core.analysis_results.DataGridDimension` items
        The dimensions.

    Returns
    -------
    dict
        A dictionary mapping the names of the variables to the dimensions found.
    """
    result_dims = {}
    varstofind = set(variables)
    for dim in dimensions:
        if not varstofind:
            break
        for var in varstofind.copy():
            if dim.variable == var:
                if isinstance(variables, collections.abc.Mapping) and (varconstraint := variables.get(var)) is not None:
                    if not isinstance(varconstraint, DimensionConstraint):
                        varconstraint = DimensionConstraint(*varconstraint)
                    varconstraint.validate(dim)
                result_dims[var] = dim
                varstofind.remove(var)
    if varstofind:
        raise RuntimeError("did not find the expected dimensions: {!r}".format(varstofind))
    return result_dims


def dtype_to_vartype(dtype: numpy.dtype | pandas.core.dtypes.base.ExtensionDtype) -> VarType:
    if dtype in [numpy.dtypes.StrDType | numpy.dtypes.StringDType | numpy.dtypes.ObjectDType]:
        return "Categorical"
    if isinstance(dtype, (pandas.StringDtype, pandas.CategoricalDtype)):
        return "Categorical"
    return "Numerical"


def vartypes_by_name_from_dataframe(data: pandas.DataFrame) -> dict[str, VarType]:
    """Get variable names and types.
    
    Parameters
    ----------
    data: pandas.DataFrame
        The dataframe in which one column contains the data for one variable.

    Returns
    -------
    dict mapping str to VarType
        A dictionary mapping the variable names to the variable types.
        The order of the variables is the same in the dictionary keys as in the dataframe columns.
    """
    return {name: dtype_to_vartype(dtype) for name, dtype in zip(data.columns, data.dtypes)}


def stats_from_analysis_report(report: khiops.core.PreparationReport, datasetinfo: DatasetInfo, jtname: str):
    """Extract stats from a Khiops analysis report.

    Parameters
    ----------
    report: khiops.core.PreparationReport
        The analysis report.
    datasetinfo: DatasetInfo
        Information about the dataset.
    jt: str
        The name of the variable that concatenates the target and the treatment together.

    Returns
    -------
    Stats
        The statistics.
    """
    jname = datasetinfo.jname
    jdim = find_dimensions({jname: DimensionConstraint("Categorical", "Value groups")}, report.get_variable_statistics(jname).data_grid.dimensions)[jname]
    js = list(chain.from_iterable(jpart.values for jpart in jdim.partition))
    tname = datasetinfo.tname
    tdim, jtdim = itemgetter(tname, jtname)(find_dimensions(
        {tname: DimensionConstraint("Categorical", "Value groups"), jtname: DimensionConstraint("Categorical", "Values")},
        report.get_variable_statistics(tname).data_grid.dimensions))
    ts = list(chain.from_iterable(tpart.values for tpart in tdim.partition))
    jts = [part.value for part in jtdim.partition]
    xnames, informative_xnames, noninformative_xnames = [], [], []
    xstats = {}
    for varname in report.get_variable_names():
        if varname in [tname, jname]:
            continue  # Skip treatment and target variables.
        xname = varname
        xnames.append(xname)
        stats = report.get_variable_statistics(xname)
        is_not_informative = stats.level == 0
        if is_not_informative:
            xstats[xname] = VarStats(False, stats.level, [], None)
            noninformative_xnames.append(xname)
            continue
        informative_xnames.append(xname)
        xdim = find_dimensions([xname], stats.data_grid.dimensions)[xname]
        is_ = xdim.partition
        xfreqs = stats.data_grid.part_target_frequencies
        xstats[xname] = VarStats(True, stats.level, is_, build_ijt_table_by_column(xname, jts, is_, lambda iindex, _: xfreqs[iindex]))
    return Stats(GeneralStats(js, ts, jts, informative_xnames, noninformative_xnames), xstats)


def compute_stats(dataset: pandas.DataFrame, datasetinfo: DatasetInfo, fileoutput: FileOutput, maxparts: int | None = None) -> tuple[Stats, UpliftDictionary]:
    logger.info("Computing stats with %r...", datasetinfo)
    
    logger.debug("Writing to data file...")
    dataset.to_csv(fileoutput.datasetfile, index=False)
    logger.debug("Done writing.")

    upliftdict = build_khiops_dict_from_dataset_file(fileoutput.dictfile, fileoutput.datasetfile, datasetinfo)

    logger.debug("Training recoder...")
    analysis_result_files = khiops.core.train_recoder(
        upliftdict.domain, upliftdict.dict.name, str(fileoutput.datasetfile), upliftdict.jtname, str(fileoutput.predictor_analysisresultfile),
        sample_percentage=100, max_trees=0, max_pairs=0, max_parts=maxparts or 0)
    logger.debug("Done training.")
    logger.debug("Analysis result files: %s, %s.", analysis_result_files[0], analysis_result_files[1])

    logger.debug("Reading analysis result file...")
    train_results = khiops.core.read_analysis_results_file(analysis_result_files[0])
    logger.debug("Done reading.")

    stats = stats_from_analysis_report(train_results.preparation_report, datasetinfo, upliftdict.jtname)

    logger.info("Done computing stats.")

    return stats, upliftdict


def group_treatments_for_variable(variable: str, datasetinfo: DatasetInfo, stats: Stats, upliftdict: UpliftDictionary, fileoutput: FileOutput, maxtreatmentgroups: int | None = None) -> VarStatsWithGroups:
    """Create groups of treatments for a variable.
    
    Create groups of treatments so that all treatments in each group give similar outcomes given the same values of the specified variable.

    Parameters
    ----------
    variable: str
        The variable on which treatment grouping will be based.
    datasetinfo: DatasetInfo
        Information about the dataset.
    stats: Stats
        The statistics computed with `compute_stats`.
    upliftdict: UpliftDictionary
        The dictionary created with `compute_stats`.
    fileoutput: FileOutput
        Paths to output files.
    maxtreatmentgroups: int or `None`
        Maximal number of treatment groups, with `None` indicating the default of Khiops.

    Returns
    -------
    VarStatsWithGroups
        Variable statistics augmented with treatment groups.
    """
    xname = variable
    xstats = stats.xstats[xname]
    if not xstats.is_informative:
        return VarStatsWithGroups(xstats, {})
    logger.info("Grouping treatments for variable %r...", xname)

    selectionvarname = add_selectionvar_to_khiops_dict(upliftdict.dict, xname, xstats.parts)
    groups_by_part = {}
    for partindex, part in enumerate(part for part in xstats.parts if not (part.part_type() == "Interval" and part.is_missing)):
        logger.debug("Grouping treatments for part %s...", part)
        partname = f"I{partindex + 1}"
        logger.debug("Training recoder...")
        analysis_result_files = khiops.core.train_recoder(
            upliftdict.domain, upliftdict.dict.name, str(fileoutput.datasetfile), datasetinfo.jname, str(fileoutput.xi_analysisresultfile(xname, partname)),
            sample_percentage=100, selection_variable=selectionvarname, selection_value=partname,
            max_trees=0, max_pairs=0, max_constructed_variables=0, max_text_features=0, max_parts=maxtreatmentgroups or 0)
        logger.debug("Done training.")
        logger.debug("Analysis result files: %s, %s.", analysis_result_files[0], analysis_result_files[1])

        logger.debug("Reading analysis result file...")
        train_results = khiops.core.read_analysis_results_file(analysis_result_files[0])
        logger.debug("Done reading.")
        logger.debug("Analysis result refers to these variable names: {%s}", ", ".join(f"'{varname}'" for varname in train_results.preparation_report.get_variable_names()))
    
        if not train_results.preparation_report.target_values:
            logger.debug("Empty preparation report.")
        group_results = train_results.preparation_report.get_variable_statistics(datasetinfo.tname)
        logger.debug("Level of treatment %r is %f.", datasetinfo.tname, group_results.level)
    
        if group_results.level == 0:  # ==> Put all treatments into the same group.
            groups_by_part[part] = (tuple(stats.generalstats.ts),)
        else:
            groups_by_part[part] = get_repaired_groups(group_results.data_grid.dimensions[0], stats)
        logger.debug("Done grouping treatments for part %s...", part)
    
    upliftdict.dict.remove_variable(selectionvarname)

    logger.info("Done grouping treatments for variable %r.", xname)
    
    return VarStatsWithGroups(xstats, groups_by_part)


def get_repaired_groups(dimension: khiops.core.DataGridDimension, stats: Stats) -> Groups:
    treatment_groups = dimension.partition
    logger.debug("Groups before repairs: %s.", tuple(tuple(grp.to_dict()) for grp in treatment_groups))
    logger.debug("Repairing groups...")
    repaired_groups = repair_groups(treatment_groups, stats.generalstats.ts)
    logger.debug("Done repairing.")
    logger.debug("Groups after repairs: %s.", repaired_groups)
    return repaired_groups


def repair_groups(groups, all_treatments):
    """Complete groups with the default values."""
    resulting_groups = []
    marked_elements = set()
    
    default_group_index_in_res = -1
    default_group_values = set()

    # 1. Analyze partition, find groups and default group.
    for i, part in enumerate(groups):
        current_group = tuple(part.values) 
        resulting_groups.append(current_group)
        marked_elements.update(current_group)

        if part.is_default_part:
            default_group_index_in_res = i
            default_group_values.update(current_group)

    # 2. Find unmarked elements.
    unmarked_elements = set(str(t) for t in all_treatments).difference(marked_elements)

    # 3. Merge unmarked elements into the default group, if any.
    if unmarked_elements:
        if default_group_index_in_res != -1:  # A default group has been found.
            # Merge unmarked elements with existing values of the default group.
            merged_group = default_group_values.union(unmarked_elements)
            # Update result with correct index.
            resulting_groups[default_group_index_in_res] = sorted(tuple(merged_group))
        else:
            # No default group found, set unmarked elements apart.
            resulting_groups.append(sorted(tuple(unmarked_elements)))

    return tuple(resulting_groups)


def build_khiops_dict_from_dataset_file(dictfilepath: Path | str, datasetfilepath: Path | str, datasetinfo: DatasetInfo) -> UpliftDictionary:
    """Build a Khiops dictionary from a dataset file.
    
    1. Read a dataset file.
    2. Create a dictionary file from the dataset.
    3. Read the dictionary file. This actually returns a dictionary domain.
    4. Get the dictionary from the dictionary domain.

    Parameters
    ----------
    dictfilepath: Path-like
        The path to the dictionary file to be created since we cannot build a dictionary in-memory.
        Also sometimes we want to inspect this file.
    datasetfilepath: Path-like
        The path to the dataset file.

    Returns
    -------
    UpliftDictionary
        A dictionary built from the dataset file.
    """
    logger.debug("Creating dictionary file from dataset file...")
    DICTNAME = "dictionary"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            r"""^Sample datasets location does not exist \([^)]*/khiops_data/samples\)\.\s+"""
            r"""Execute the kh-download-datasets script or the khiops\.tools\.download_datasets function to download them\.$""",
            UserWarning, f"^{khiops.core.internals.runner.__name__}$")
        khiops.core.build_dictionary_from_data_table(str(datasetfilepath), DICTNAME, str(dictfilepath))
    logger.debug("Done creating dictionary file.")
    logger.debug("Reading from dictionary file...")
    domain = khiops.core.read_dictionary_file(str(dictfilepath))
    logger.debug("Done reading.")
    dictionary = domain.get_dictionary(DICTNAME)
    fix_vartypes_in_khiops_dict(dictionary, datasetinfo)
    jtname = add_jtvar_to_khiops_dict(dictionary, datasetinfo)
    return UpliftDictionary(domain, dictionary, jtname)


def fix_vartypes_in_khiops_dict(dictionary: khiops.core.Dictionary, datasetinfo: DatasetInfo) -> None:
    """Set the types of categorical variables, treatment variable and target variable to "Categorical"."""
    logger.debug("Fixing variable types in Khiops dictionary...")
    for var in datasetinfo.categorical_xs:
        dictionary.get_variable(var).type = "Categorical"
    dictionary.get_variable(datasetinfo.tname).type = "Categorical"
    dictionary.get_variable(datasetinfo.jname).type = "Categorical"
    logger.debug("Done fixing variable types.")


def add_jtvar_to_khiops_dict(dictionary: khiops.core.Dictionary, datasetinfo: DatasetInfo) -> str:
    logger.debug("Adding target|treatment calculated variable to Khiops dictionary...")
    jtname = random_varname(dictionary, "j|t---")
    jtvar = khiops.core.Variable()
    jtvar.name = jtname
    jtvar.type = "Categorical"
    jtvar.used = True
    jtvar.rule = """Concat({jname},"|",{tname})""".format(jname=datasetinfo.jname, tname=datasetinfo.tname)
    dictionary.add_variable(jtvar)
    logger.debug("Done adding target|treatment calculated variable.")
    return jtname


def add_selectionvar_to_khiops_dict(dictionary: khiops.core.Dictionary, xname: str, xparts: list) -> str:
    logger.debug("Adding selection variable for variable %r to Khiops dictionary...", xname)
    selectionvarname = random_varname(dictionary, f"Selection_{xname}---")
    selectionvar = khiops.core.Variable()
    selectionvar.name = selectionvarname
    selectionvar.type = "Categorical"
    selectionvar.used = True
    selectionvar.rule = str(partition_to_rule(xparts, dictionary.get_variable(xname)))
    logger.debug("Selection variable rule: %r.", selectionvar.rule)
    dictionary.add_variable(selectionvar)
    logger.debug("Done adding selection variable.")
    return selectionvarname