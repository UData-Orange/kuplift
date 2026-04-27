from dataclasses import dataclass
import collections.abc
import typing
from itertools import chain
from operator import itemgetter, add
import pandas
import khiops.core
from .typealiases import Part, VarType
from .utils import DatasetInfo, build_table_by_row


@dataclass(frozen=True)
class VarStats:
    """Statistics for a variable.

    Attributes
    ----------
    type: VarType
        The type of the variable.
    is_informative: bool
        `True` if the variable is informative, `False` otherwise.

    level: float
        The level. Zero for a non-informative variable.

    parts: list[Part]
        The list of parts. Empty for a non-informative variable.

    nijt: pandas.DataFrame | None
        A DataFrame that is the N_ijt table of the variable if it is informative, `None` otherwise.
    """
    type: VarType
    is_informative: bool
    level: float
    parts: list[Part]
    nijt: pandas.DataFrame | None


Model = dict[str, VarStats]


@dataclass(frozen=True)
class GeneralStats:
    """Statistics mainly useful to get the list of informative variables.

    Attributes
    ----------
    js: list of str
        The list of targets.

    ts: list of str
        The list of treatments.

    jts: list of str
        The list of target-treatment pairs.

    informative_xnames: list of str
        The list of informative input variables.

    noninformative_xnames: list of str
        The list of non-informative input variables.
    """
    js: list[str]
    ts: list[str]
    jts: list[str]
    informative_xnames: list[str]
    noninformative_xnames: list[str]


@dataclass(frozen=True)
class Stats(GeneralStats):
    """Statistics.

    Attributes
    ----------
    generalstats: GeneralStats
        Statistics not including input-variable specific stats.

    model: Model
        The discretization model.
    """
    model: Model


def stats_from_analysis_report(report: khiops.core.PreparationReport, datasetinfo: DatasetInfo, jtname: str) -> Stats:
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
            xstats[xname] = VarStats(stats.type, False, stats.level, [], None)
            noninformative_xnames.append(xname)
            continue
        informative_xnames.append(xname)
        xdim = find_dimensions([xname], stats.data_grid.dimensions)[xname]
        is_, xfreqs = merge_missing_into_first_interval(xdim.partition, stats.data_grid.part_target_frequencies)
        xstats[xname] = VarStats(stats.type, True, stats.level, is_, build_table_by_row(is_, jts, lambda iindex, _: xfreqs[iindex]))
    return Stats(js, ts, jts, informative_xnames, noninformative_xnames, xstats)


def merge_missing_into_first_interval(parts: list[Part], xfreqs: list[list[int]]) -> list[list[int]]:
    if parts[0].part_type() == "Interval" and parts[0].is_missing and len(parts) > 1:
        return parts[1:], [list(map(add, xfreqs[0], xfreqs[1])), *xfreqs[2:]]
    else:
        return parts, xfreqs
    

@dataclass(frozen=True)
class DimensionConstraint:
    """Constraint the query of a dimension by its type and partition type."""
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