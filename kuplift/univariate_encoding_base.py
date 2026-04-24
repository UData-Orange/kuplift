from abc import ABC, abstractmethod
import collections.abc
import pandas
from .preparation_report import Model, Stats
from .utils import transform_variable, split_jt, join_jt, probabilities, build_table_by_cell
from .typealiases import Partition


def transform(model: Model, data: pandas.DataFrame, variables_to_transform: list) -> pandas.DataFrame:
    return data[variables_to_transform].transform(lambda column: transform_variable(model[column.name].parts, column))


def get_level_of_var(model: Model, variable: str) -> float:
    return model[variable].level


def get_levels(model: Model) -> list[tuple[str, float]]:
    return sorted(((varname, varstats.level) for varname, varstats in model.items()), key=lambda name_level_pair: (-name_level_pair[1], name_level_pair[0]))


def get_partition_of_var(model: Model, variable: str) -> Partition:
    return model[variable].parts


def get_target_frequencies_of_var(model: Model, variable: str) -> pandas.DataFrame:
    return model[variable].nijt


def get_target_probabilities_of_var(model: Model, variable: str, target_modalities: tuple[str, str], jts: collections.abc.Sequence[str]) -> pandas.DataFrame:
    frequencies = get_target_frequencies_of_var(model, variable)
    def cell(iindex, i, jtindex, jt):
        j, t = split_jt(jt)
        return probabilities(frequencies[i][jt], frequencies[i][join_jt(get_other_target_modality(target_modalities, j), t)])
    return build_table_by_cell(get_partition_of_var(model, variable), jts, cell)


def get_uplift_of_var(model: Model, variable: str, target_modalities: tuple[str, str], ts: collections.abc.Sequence[str], jts: collections.abc.Sequence[str], successvalue: str, reftreatment: str) -> pandas.DataFrame:
    probabilities = get_target_probabilities_of_var(model, variable, target_modalities, jts)
    def cell(iindex, i, tindex, t):
        return probabilities[i][join_jt(successvalue, t)] - probabilities[i][join_jt(successvalue, reftreatment)]
    return build_table_by_cell(get_partition_of_var(model, variable), ts, cell)


def get_other_target_modality(target_modalities: tuple[str, str], target_modality: str) -> str:
    if target_modalities[0] == target_modality:
        return target_modalities[1]
    else:
        return target_modalities[0]
    

class UnivariateEncodingBase(ABC):
    """
    Base class to construct univariate encoding classes implementing various fitting algorithms.
    The derived classes must implement the fit() method that sets the attributes so that the properties and methods work properly.
    Attributes to be set by the fit() method:
    - _variable_cols: the columns, from the dataset, corresponding to the variables;
    - _treatment_col: the column, from the dataset, corresponding to the treatment;
    - _target_col: the column, from the dataset, corresponding to the target.
    - _stats: the statistics as a Stats object.
    - _fit_performed: to be set to True when fit() succeeds.
    """


    def __init__(self):
        self._variable_cols: pandas.DataFrame | None = None
        self._treatment_col: pandas.Series | None = None
        self._target_col: pandas.Series | None = None
        self._stats: Stats | None = None
        self._fit_performed: bool = False


    @abstractmethod
    def fit():
        pass


    def _check_fit_performed(self) -> None:
        if not self._fit_performed:
            raise RuntimeError("Fit operation not performed yet.")


    def fit_transform(self, data: pandas.DataFrame, *args) -> pandas.DataFrame:
        """Learn a discretization model and transform the data.
        
        Parameters
        ----------
        data: pandas.DataFrame
            Dataframe containing feature variables. Categorical variables must have a string, categorical or object dtype to avoid beeing processed as numerical variables.
        args:
            Same parameters as `fit()`, without the first one which is already in `data`.

        Returns
        -------
        pandas.Dataframe
            Pandas Dataframe that contains encoded data.
        """
        self.fit(data, *args)
        return self.transform(data)


    @property
    def variable_columns(self) -> pandas.DataFrame:
        """The data columns of all variables.
        
        This means all the data from the dataset but the treatment and target columns."""
        return self._variable_cols
    

    @property
    def treatment_column(self) -> pandas.Series:
        """The treatment column from the dataset."""
        return self._treatment_col
    

    @property
    def target_column(self) -> pandas.Series:
        """The target column from the dataset."""
        return self._target_col


    @property
    def _model(self) -> Model:
        self._check_fit_performed()
        return self._stats.model

    
    @property
    def input_variables(self) -> list[str]:
        """The names of the variables."""
        self._check_fit_performed()
        return list(self._model)


    @property
    def informative_input_variables(self) -> list[str]:
        """The names of the informative variables."""
        self._check_fit_performed()
        return self._stats.informative_xnames


    @property
    def noninformative_input_variables(self) -> list[str]:
        """The names of the non-informative variables."""
        self._check_fit_performed()
        return self._stats.noninformative_xnames


    @property
    def treatment_name(self) -> str:
        """The name of the treatment column."""
        self._check_fit_performed()
        return self._treatment_col.name


    @property
    def target_name(self) -> str:
        """The name of the target column."""
        self._check_fit_performed()
        return self._target_col.name


    @property
    def treatment_modalities(self) -> list:
        """All the different treatments from the dataset."""
        self._check_fit_performed()
        return self._stats.ts


    @property
    def target_modalities(self) -> list:
        """All the different targets from the dataset."""
        self._check_fit_performed()
        return self._stats.js


    @property
    def target_treatment_pairs(self) -> list[str]:
        """All (target, treatment) pairs as a list of "target|treatment"-formatted strings."""
        self._check_fit_performed()
        return self._stats.jts
    

    def transform(self, data):
        """Apply the discretization model learned by the fit() method.

        Parameters
        ----------
        data: pandas.DataFrame
            Dataframe containing feature variables.

        Returns
        -------
        pandas.DataFrame
            Pandas Dataframe that contains encoded data.
        """
        self._check_fit_performed()
        return transform(self._model, data, self.informative_input_variables)


    def get_levels(self):
        """Get the level of all variables.

        Returns
        -------
        list[tuple[str, float]]
            (variable-name, variable-level) pairs in decreasing level order.
        """
        self._check_fit_performed()
        return get_levels(self._model)

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
        self._check_fit_performed()
        return get_level_of_var(self._model, variable)


    def get_partition(self, variable: str) -> Partition:
        """Get the partition corresponding to a single variable of the model.

        Parameters
        ----------
        variable: str
            The variable name.

        Returns
        -------
        Partition
            The partition corresponding to a single variable of the model.
        """
        self._check_fit_performed()
        return get_partition_of_var(self._model, variable)


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
        self._check_fit_performed()
        return get_target_frequencies_of_var(self._model, variable)


    def get_target_probabilities(self, variable):
        """Get the probabilities P_ijt for a variable.

        Parameters
        ----------
        variable: str
            The variable name.

        Returns
        -------
        pandas.DataFrame
            The probability table for the variable.
        """
        self._check_fit_performed()
        return get_target_probabilities_of_var(self._model, variable, self._stats.js, self._stats.jts)


    def get_target_probabilities_of_treatment_groups(self, variable):
        """Get the probabilities P_ijg for a variable.

        Parameters
        ----------
        variable: str
            The variable name.

        Returns
        -------
        pandas.DataFrame
            The probability table for the variable.
        """
        self._check_fit_performed()
        return get_target_probabilities_of_var(self._model, variable, self._stats.js, self._stats.jts)


    def get_uplift(self, successvalue, reftreatment, variable):
        """Get the uplift Uplift_it for a variable.

        Parameters
        ----------
        successvalue
            The success value of the target.
        reftreatment
            The reference treatment to which all the other treatments are compared.
        variable: str
            The name of the variable.

        Returns
        -------
        pandas.DataFrame
            The uplift table for the variable.
        """
        self._check_fit_performed()
        return get_uplift_of_var(self._model, variable, self._stats.js, self._stats.ts, self._stats.jts, successvalue, reftreatment)