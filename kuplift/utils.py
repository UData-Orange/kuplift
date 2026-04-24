from dataclasses import dataclass
import collections.abc
from math import isnan
import numpy
import pandas
import pandas.core.dtypes.base
import khiops.core
from .typealiases import VarType, Partition, NumVarPartition, CatVarPartition


def build_table_by_cell(columns: collections.abc.Sequence, rows: collections.abc.Sequence, func: collections.abc.Callable) -> pandas.DataFrame:
    return pandas.DataFrame(index=rows, data={col: [func(colindex, col, rowindex, row) for rowindex, row in enumerate(rows)] for colindex, col in enumerate(columns)})


def build_table_by_column(columns: collections.abc.Sequence, rows: collections.abc.Sequence, func: collections.abc.Callable) -> pandas.DataFrame:
    return pandas.DataFrame(index=rows, data={col: func(colindex, col) for colindex, col in enumerate(columns)})


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
class Dictionary:
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


def in_interval(value: float, interval: khiops.core.PartInterval) -> bool:
    return (
        interval.is_left_open and interval.is_right_open
        or interval.is_left_open and value <= interval.upper_bound
        or interval.lower_bound < value and interval.is_right_open
        or interval.lower_bound < value <= interval.upper_bound
    )


def transform_variable(parts: Partition, values: pandas.Series) -> int:
    if not parts:
        raise ValueError("No parts.")
    match parts[0].part_type():
        case "Interval":
            return transform_numerical_variable(parts, values)
        case "Value group":
            return transform_categorical_variable(parts, values)
        case unsupported:
            raise ValueError("Unsupported {!r} part type.".format(unsupported))


def transform_numerical_variable(parts: NumVarPartition, values: pandas.Series) -> int:
    def transform_value(value):
        if not isinstance(value, (int, float)) or isnan(value):
            return 0
        else:
            return next(interval_index for interval_index, interval in enumerate(parts) if in_interval(value, interval))
    return values.transform(transform_value)


def transform_categorical_variable(parts: CatVarPartition, values: pandas.Series) -> int:
    default_group_index = next(group_index for group_index, group in enumerate(parts) if group.is_default_part)
    def transform_value(value):
        for group_index, group in enumerate(parts):
            if value in group.values:
                return group_index
        return default_group_index
    return values.transform(transform_value)


def split_jt(jt: str) -> tuple[str, str]:
    """Get j and t from j|t."""
    return tuple(jt.split("|"))


def join_jt(j: str, t: str) -> str:
    """Get j|t from j and t."""
    return "{}|{}".format(j, t)


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


def fix_valuegroups(valuegroups: list[khiops.core.PartValueGroup], all_values: collections.abc.Iterable[str]) -> None:
    """Is it still needed with Khiops V11?"""
    unvisited_values = set(all_values)
    default_valuegroup = None
    for valuegroup in valuegroups:
        unvisited_values.difference_update(valuegroup.values)
        if valuegroup.is_default_part:
            default_valuegroup = valuegroup
    if unvisited_values:
        if default_valuegroup is None:
            new_default_valuegroup = khiops.core.PartValueGroup(sorted(list(unvisited_values)))
            new_default_valuegroup.is_default_part = True
            valuegroups.append(new_default_valuegroup)
        else:
            default_valuegroup.values.extend(unvisited_values)
            default_valuegroup.values.sort()


def probabilities(frequency, *other_frequencies):
    return frequency / (frequency + sum(other_frequencies))