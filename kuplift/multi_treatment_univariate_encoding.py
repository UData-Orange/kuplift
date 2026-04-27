# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

"""Multi-treatment Univariate Encoding

This module contains everything needed to make univariate variable transformation
capable of merging treatments that give similar outcome.
"""

from pathlib import Path
from .helperfunctions import partition_to_rule, random_varname
from tempfile import TemporaryDirectory
import logging
from functools import cached_property
from dataclasses import dataclass
import warnings
import khiops.core
import pandas
from .utils import DatasetInfo, Dictionary, fix_valuegroups
from .treatment_grouping import StatsWithGroups, VarStatsWithGroups
from .preparation_report import Stats, stats_from_analysis_report
from .univariate_encoding_treatment_grouping import UnivariateEncodingWithGroupsBase

logger = logging.getLogger(__name__)


class MultiTreatmentUnivariateEncoding(UnivariateEncodingWithGroupsBase):
    """
    The MultiTreatmentUnivariateEncoding class makes use of the khiops Python wrapper
    and enables one to fit and transform data while grouping treatments giving similar
    outcome.
    """
    def fit(self, data: pandas.DataFrame, treatment_col: pandas.Series, target_col: pandas.Series, maxparts: int | None = None, maxtreatmentgroups: int | None = None, outputdir: Path | str | None = None) -> None:
        """Learn a discretization model using Khiops.

        Parameters
        ----------
        data: pandas.DataFrame
            Dataframe containing feature variables. Categorical variables must have a string, categorical or object dtype to avoid beeing processed as numerical variables.
        treatment_col: pandas.Series
            Treatment column.
        target_col: pandas.Series
            Outcome column.
        maxparts: int, default=None
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
        datasetinfo = DatasetInfo(target_col.name, treatment_col.name, data.columns.to_list(), len(dataset))

        stats, upliftdict = compute_stats(dataset, datasetinfo, fileoutput, maxparts)

        # Disable all input variables since we will create a selection variable for each one in turn
        # and it is that selection variable that will be enabled.
        for xname in stats.model:
            upliftdict.dict.get_variable(xname).used = False

        xstats_with_groups = {}
        for xname in stats.model:
            xstats_with_groups[xname] = group_treatments_for_variable(xname, datasetinfo, stats, upliftdict, fileoutput, maxtreatmentgroups)

        self._variable_cols = data
        self._treatment_col = treatment_col
        self._target_col = target_col
        self._stats = StatsWithGroups(stats.js, stats.ts, stats.jts, stats.informative_xnames, stats.noninformative_xnames, xstats_with_groups)
        self._fit_performed = True


@dataclass(frozen=True)
class FileOutput:
    """Compute paths to files and directories to be created."""
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


def compute_stats(dataset: pandas.DataFrame, datasetinfo: DatasetInfo, fileoutput: FileOutput, maxparts: int | None = None) -> tuple[Stats, Dictionary]:
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


def group_treatments_for_variable(variable: str, datasetinfo: DatasetInfo, stats: Stats, upliftdict: Dictionary, fileoutput: FileOutput, maxtreatmentgroups: int | None = None) -> VarStatsWithGroups:
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
    upliftdict: Dictionary
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
    xstats = stats.model[xname]
    if not xstats.is_informative:
        logger.info("Variable %r is not informative -> skipped treatment grouping.", xname)
        return VarStatsWithGroups(xstats.type, xstats.is_informative, xstats.level, xstats.parts, xstats.nijt, {}, {})
    logger.info("Grouping treatments for variable %r...", xname)

    selectionvarname = add_selectionvar_to_khiops_dict(upliftdict.dict, xname, xstats.parts)
    groups_by_parts = {}
    groups_by_treatments_by_parts = {}
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
        logger.debug("Level of treatment %r is %f%s.", datasetinfo.tname, group_results.level, " -> put all treatments into the same group" if group_results.level == 0 else "")

        if group_results.level == 0:  # ==> Put all treatments into the same group.
            group = khiops.core.PartValueGroup(stats.ts)
            group.is_default_part = True
            groups_by_parts[part] = [group]
            groups_by_treatments_by_parts[part] = {t: groups_by_parts[part][0] for t in stats.ts}
        else:
            groups = group_results.data_grid.dimensions[0].partition
            logger.debug("Groups before fix: %s.", ", ".join(str(grp) for grp in groups))
            fix_valuegroups(groups, stats.ts)
            logger.debug("Groups after fix: %s.", ", ".join(str(grp) for grp in groups))
            groups_by_parts[part] = groups
            groups_by_treatments_by_parts[part] = {t: group for group in groups for t in group.values}
        logger.debug("Done grouping treatments for part %s...", part)

    upliftdict.dict.remove_variable(selectionvarname)

    logger.info("Done grouping treatments for variable %r.", xname)

    return VarStatsWithGroups(xstats.type, xstats.is_informative, xstats.level, xstats.parts, xstats.nijt, groups_by_parts, groups_by_treatments_by_parts)


def build_khiops_dict_from_dataset_file(dictfilepath: Path | str, datasetfilepath: Path | str, datasetinfo: DatasetInfo) -> Dictionary:
    """Build a Khiops dictionary from a dataset file.

    1. Read a dataset file.
    2. Create a dictionary file from the dataset.
    3. Read the dictionary file. This actually returns a dictionary domain.
    4. Get the dictionary from the dictionary domain.
    5. Fix the types of the variables in the dictionary.
    6. Add a j|t calculated variable in the dictionary.

    Parameters
    ----------
    dictfilepath: Path-like
        The path to the dictionary file to be created since we cannot build a dictionary in-memory.
        Also sometimes we want to inspect this file.
    datasetfilepath: Path-like
        The path to the dataset file.

    Returns
    -------
    Dictionary
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
    check_vartypes_in_khiops_dict(dictionary, datasetinfo)
    jtname = add_jtvar_to_khiops_dict(dictionary, datasetinfo)
    return Dictionary(domain, dictionary, jtname)


def fix_vartypes_in_khiops_dict(dictionary: khiops.core.Dictionary, datasetinfo: DatasetInfo) -> None:
    """Set types of treatment and target variables to "Categorical"."""
    logger.debug("Fixing variable types in Khiops dictionary...")
    if dictionary.get_variable(datasetinfo.tname).type != "Categorical":
        warnings.warn("Treatment variable not detected as categorical; fixing...")
        dictionary.get_variable(datasetinfo.tname).type = "Categorical"
    if dictionary.get_variable(datasetinfo.jname).type != "Categorical":
        warnings.warn("Target variable not detected as categorical; fixing...")
        dictionary.get_variable(datasetinfo.jname).type = "Categorical"
    logger.debug("Done fixing variable types.")


def check_vartypes_in_khiops_dict(dictionary: khiops.core.Dictionary, datasetinfo: DatasetInfo) -> None:
    """Check that all input variables are either numerical or categorical."""
    for x in datasetinfo.xs:
        vartype = dictionary.get_variable(x).type
        logger.debug("Type of input variable %r is %r.", x, vartype)
        if vartype not in ["Numerical", "Categorical"]:
            raise ValueError("unsupported variable type {}".format(vartype))


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