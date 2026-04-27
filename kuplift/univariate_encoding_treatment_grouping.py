import collections.abc
import pandas
from .treatment_grouping import ModelWithGroups, TreatmentGroups
from .univariate_encoding_base import get_target_frequencies_of_var, get_other_target_modality, get_partition_of_var, UnivariateEncodingBase
from .utils import split_jt, join_jt, probabilities, build_table_by_cell
from .typealiases import Part


def get_treatment_groups_of_var(model: ModelWithGroups, variable: str) -> dict[Part, TreatmentGroups]:
    return model[variable].groups_by_parts


def get_treatment_groups(model: ModelWithGroups) -> dict[str, dict[Part, TreatmentGroups]]:
    return {varname: varstats.groups_by_parts for varname, varstats in model.items()}


def get_target_probabilities_of_var_with_treatment_groups(model: ModelWithGroups, variable: str, target_modalities: tuple[str, str], jts: collections.abc.Sequence[str]) -> pandas.DataFrame:
    frequencies = get_target_frequencies_of_var(model, variable)
    def get_frequencies_of_group(i, j, t):
        return sum(frequencies[join_jt(j, treatment)][i] for treatment in model[variable].groups_by_treatments_by_parts[i][t].values)
    def cell(iindex, i, jtindex, jt):
        j, t = split_jt(jt)
        return probabilities(get_frequencies_of_group(i, j, t), get_frequencies_of_group(i, get_other_target_modality(target_modalities, j), t))
    return build_table_by_cell(get_partition_of_var(model, variable), jts, cell)


def get_uplift_of_var_with_treatment_groups(model: ModelWithGroups, variable: str, target_modalities: tuple[str, str], ts: collections.abc.Sequence[str], jts: collections.abc.Sequence[str], successvalue: str, reftreatment: str) -> pandas.DataFrame:
    probabilities = get_target_probabilities_of_var_with_treatment_groups(model, variable, target_modalities, jts)
    def cell(iindex, i, tindex, t):
        return probabilities[join_jt(successvalue, t)][i] - probabilities[join_jt(successvalue, reftreatment)][i]
    return build_table_by_cell(get_partition_of_var(model, variable), ts, cell)


class UnivariateEncodingWithGroupsBase(UnivariateEncodingBase):
    def get_treatment_groups(self, variable: str | None = None) -> dict[Part, TreatmentGroups] | dict[str, dict[Part, TreatmentGroups]]:
        """Get the groups of treatments for one or all variables.

        Parameters
        ----------
        variable: str | None
            If set to None, get groups of all variables, otherwise get groups of specified variable.

        Returns
        -------
        If `variable` is None, returns a dict mapping variable names to dictionaries mapping parts to treatment groups.
        If `variable` is not None, returns a dict mapping parts to treatment groups.
        """
        self._check_fit_performed()
        if variable is None:
            return get_treatment_groups(self._model)
        else:
            return get_treatment_groups_of_var(self._model, variable)
        

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
        return get_target_probabilities_of_var_with_treatment_groups(self._model, variable, tuple(self._stats.js), self._stats.jts)


    def get_uplift_of_treatment_groups(self, successvalue, reftreatment, variable):
        """Get the uplift Uplift_ig for a variable.

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
        return get_uplift_of_var_with_treatment_groups(self._model, variable, tuple(self._stats.js), self._stats.ts, self._stats.jts, successvalue, reftreatment)