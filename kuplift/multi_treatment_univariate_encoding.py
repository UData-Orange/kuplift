# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

"""Multi-treatment Univariate Encoding

This module contains everything needed to make univariate variable transformation
capable of merging treatments that give similar outcome.

The main class of this module is 'MultiTreatmentUnivariateEncoding'.
"""

from pathlib import Path
from .helperclasses import ValGrp, ValGrpPartition, Interval, IntervalPartition, TargetTreatmentPair, TargetTreatmentGroupPair
from .helperfunctions import partition_to_rule
from tempfile import TemporaryDirectory
import logging

logger = logging.getLogger(__name__)


class MultiTreatmentUnivariateEncoding:
    """
    The MultiTreatmentUnivariateEncoding class makes use of the khiops Python wrapper
    and enables one to fit and transform data while grouping treatments giving similar
    outcome.
    
    Attributes
    ----------
    model: dict mapping str to ValGrpPartition or IntervalPartition
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

    treatment_groups: 
        A dict mapping variable names to dictionaries mapping parts to treatment groups.
        For instance:
            {
                "var1": { Interval(-inf, 0.2): ( ("T0", "T1), ("T2", "T3", "T4)       ),
                          Interval( 0.2, inf): ( ("T5", "T6"), ("T7", "T8")           ) }
                "var2": { ValGrp(["A", "B"]) : ( ("T0", "T1", "T2")                   ),
                          ValGrp(["C", "D"]) : ( ("T3", "T4", "T5", "T6", "T7", "T8") ) }
            }
    """

    def __init__(self):
        self.model: dict[str, ValGrpPartition | IntervalPartition] = {}
        self.levels: list[tuple[str, float]] = []
        self.variable_cols = None
        self.treatment_col = None
        self.target_col = None
        self.treatment_groups = None


    @property
    def input_variables(self):
        """list of str
        
        The names of the variables.
        """
        return self.variable_cols.columns.to_list()
    

    @property
    def informative_input_variables(self):
        """list of str
        
        The names of the informative variables.
        """
        return [v for v in self.input_variables if v in self.model]
    

    @property
    def noninformative_input_variables(self):
        """list of str
        
        The names of the non-informative variables.
        """
        return [v for v in self.input_variables if v not in self.model]
    

    @property
    def treatment_name(self):
        """str
        
        The name of the treatment column.
        """
        return self.treatment_col.name
    

    @property
    def target_name(self):
        """str
        
        The name of the target column.
        """
        return self.target_col.name
    

    @property
    def treatment_modalities(self):
        """list
        
        All the different treatments from the dataset.
        """
        return list(self.treatment_col.unique())
    

    @property
    def target_modalities(self):
        """list
        
        All the different targets from the dataset.
        """
        return list(self.target_col.unique())
    

    @property
    def target_treatment_pairs(self):
        """list of TargetTreatmentPair
        
        All (target, treatment) pairs.
        """
        return [TargetTreatmentPair(target, treatment) for target in self.target_modalities for treatment in self.treatment_modalities]
    

    def fit(self, data, treatment_col, target_col, *, maxpartnumber = None, maxtreatmentgroups = None, outputdir = None):
        """Learn a discretisation model using Khiops.

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
            The maximal number of intervals or groups. None means default to the 'khiops' program default.
        maxtreatmentgroups: int, default=None
            The maximal number of groups to define when grouping treatments together. None means automatic.
        outputdir: Path-like
            Set this if you want khiops's workfiles to be kept in a specific directory. If None, fallback
            to the default behaviour which is to have khiops write its files into a temporary directory that
            is deleted when the work is done.
        """
        model, treatment_groups, levels = uplift_MODL(data, treatment_col, target_col, maxpartnumber=maxpartnumber, maxtreatmentgroups=maxtreatmentgroups, outputdir=outputdir)

        # Only write to the instance's attributes if all the above succeeded.
        self.model = model
        self.levels = levels
        self.variable_cols = data
        self.treatment_col = treatment_col
        self.target_col = target_col
        self.treatment_groups = treatment_groups


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

    
    def fit_transform(self, data, treatment_col, target_col, *, maxpartnumber = None, maxtreatmentgroups = None, outputdir = None):
        """Learn a discretisation model using Khiops and transform the data.

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
            The maximal number of intervals or groups. None means default to the 'khiops' program default.
        maxtreatmentgroups: int, default=None
            The maximal number of groups to define when grouping treatments together. None means automatic.
        outputdir: Path-like
            Set this if you want khiops's workfiles to be kept in a specific directory. If None, fallback
            to the default behaviour which is to have khiops write its files into a temporary directory that
            is deleted when the work is done.

        Returns
        -------
        pd.Dataframe
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
        return self.levels
    

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
        return dict(self.levels)[variable]
    

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
    

    def get_treatment_groups(self, variable=None):
        """Get the groups of treatments for one or all variables.

        Parameters
        ----------
        variable
            If set to None, get groups of all variables, otherwise get groups of specified variable.

        Returns
        -------
        If `variable` is not None, returns a dict mapping parts to treatment groups.
        If `variable` is None, returns a dict mapping variable names to dictionaries mapping parts to treatment groups.
        """
        return self.treatment_groups if variable is None else self.treatment_groups[variable]
    

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


    def get_target_frequencies_of_treatment_groups(self, variable):
        """Get the frequencies for each (target, treatment group) pair, for each part in the variable's partition.
        
        The frequencies are computed for a single variable.
        
        Parameters
        ----------
        variable: str
            The variable name.

        Returns
        -------
        dict[Part, pd.Series]
            The frequencies as a dict mapping parts to Series which index represents target-treatmentgroup pairs and which values are the frequencies.
        """
        varcol = self.variable_cols[variable]
        partition = self.get_partition(variable)
        return {
            part: pd.Series(
                {
                    TargetTreatmentGroupPair(target, treatmentgrp): len(
                        varcol[
                            (self.treatment_col.isin(treatmentgrp)) & (self.target_col == target) & varcol.map(lambda elem: partition.transform_elem(elem) == i)
                        ]
                    )
                    for target in self.target_modalities
                    for treatmentgrp in self.treatment_groups[variable][part]
                }
            )
            for i, part in enumerate(partition)
        }
    

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
        freqs = self.get_target_frequencies(variable)
        return freqs["Part"].to_frame().join(
            pd.DataFrame({
                ttpair: [
                    freqs[ttpair][freqs["Part"] == part].sum() / freqs[
                        [TargetTreatmentPair(target, ttpair.treatment) for target in self.target_modalities]
                    ][freqs["Part"] == part].sum().sum()
                    for part in self.get_partition(variable)
                ]
                for ttpair in self.target_treatment_pairs
            })
        )
    

    def get_target_probabilities_of_treatment_groups(self, variable):
        """Get the probabilities P(target|treatment group) for each (target, treatment group) pair.
        
        The probabilities are computed for a single variable.
        
        Parameters
        ----------
        variable: str
            The variable name.

        Returns
        -------
        dict[Part, pd.Series]
            The probabilities as a dict mapping parts to Series which index represents target-treatmentgroup pairs and which values are the probabilities.
        """
        freqs = self.get_target_frequencies_of_treatment_groups(variable)
        return {
            part: pd.Series({
                ttgrppair: partfreqs[ttgrppair] / partfreqs[partfreqs.index.map(lambda i: i.treatment_group == ttgrppair.treatment_group)].sum()
                for ttgrppair in partfreqs.index
            })
            for part, partfreqs in freqs.items()
        }
    

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
        if reftreatment not in self.treatment_modalities:
            raise ValueError("treatment %r not in known treatments {%s}" % (reftreatment, ", ".join(f"'{t}'" for t in self.treatment_modalities)))
        if reftarget not in self.target_modalities:
            raise ValueError("target %r not in known targets {%s}" % (reftarget, ", ".join(f"'{y}'" for y in self.target_modalities)))
        probs = self.get_target_probabilities(variable)
        refprobs = probs[TargetTreatmentPair(reftarget, reftreatment)]
        return probs["Part"].to_frame().join(pd.DataFrame({
            f"Uplift {reftarget} {treatment}": probs[TargetTreatmentPair(reftarget, treatment)] - refprobs
            for treatment in self.treatment_modalities if treatment != reftreatment
        }))
    

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

        dict[Part, pd.Series]
            The probabilities as a dict mapping parts to Series which index represents target-treatmentgroup pairs and
            which values are the differences P(reftarget|treatment group) - P(reftarget|reftreatment).
        """
        if reftreatment not in self.treatment_modalities:
            raise ValueError("treatment %r not in known treatments {%s}" % (reftreatment, ", ".join(f"'{t}'" for t in self.treatment_modalities)))
        if reftarget not in self.target_modalities:
            raise ValueError("target %r not in known targets {%s}" % (reftarget, ", ".join(f"'{y}'" for y in self.target_modalities)))
        probs = self.get_target_probabilities(variable)
        refttpair = TargetTreatmentPair(reftarget, reftreatment)
        refprobs = probs[["Part", refttpair]]
        tgrp_probs = self.get_target_probabilities_of_treatment_groups(variable)
        return {
            part: pd.Series({
                ttgrppair: tgrp_probs[part][TargetTreatmentGroupPair(reftarget, ttgrppair.treatment_group)] - next(iter(refprobs[refprobs["Part"] == part][refttpair]))
                for ttgrppair in partprobs.index
            })
            for part, partprobs in tgrp_probs.items()
        }


#############################################################################
# Computation classes and functions from this point till the end of the file.
# This has been copied-pasted here and 10~15 dead lines have been removed but
# much more work is needed.


import pandas as pd
import numpy as np
# from scipy.special import gammaln
from khiops import core as kh
from .KWStat import *
# from sklearn.base import clone
# from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.base import BaseEstimator, TransformerMixin
import warnings


# class SingleClassPredictor(BaseEstimator, ClassifierMixin):
#     def fit(self, X, y=None):
#         self.unique_class_ = np.unique(y)[0]
#         self.classes_ = np.array([0, 1])  # Explicitly state classes for consistency
#         return self


#     def predict(self, X):
#         return np.full(X.shape[0], self.unique_class_)


#     def predict_proba(self, X):
#         probabilities = np.zeros((X.shape[0], 2)) 
        
#         # Determine the index of the unique class (0 or 1)
#         class_index = int(self.unique_class_) 
        
#         # Set the probabilities for the unique class to 1.0
#         probabilities[:, class_index] = 1.0
        
#         return probabilities


def uplift_MODL(data, treatment_col, target_col, *, maxpartnumber, maxtreatmentgroups, outputdir):
    t, y, xs = treatment_col.name, target_col.name, data.columns
    logger.info("Computing uplift with %d lines of data, %d variables, treatment column %r and target column %r...",
                 len(data), len(xs), t, y)
    all_treatments = np.sort(treatment_col.unique())

    categorical_vars = {var for var in data if data.dtypes[var] == object}
    logger.debug("Numerical variables: {%s}.", ", ".join("%r" % v for v in sorted(set(data.columns) - categorical_vars)))
    logger.debug("Categorical variables: {%s}.", ", ".join("%r" % v for v in sorted(categorical_vars)))
    
    if outputdir is None:
        tmpdir = TemporaryDirectory()
        dirpath = Path(tmpdir.name)
    else:
        dirpath = Path(outputdir)
    logger.debug("Temporary file output will be in %r.", dirpath)
    
    datatable_path = dirpath / "data.csv"
    logger.debug("Data file name: %s.", datatable_path)
    datatable_filename = str(datatable_path)
    dct_filepath = dirpath / "dictionary.kdic"
    logger.debug("Dictionary file name: %s.", dct_filepath)
    analysis_result_dirpath = dirpath / "analyse_uplift"
    logger.debug("Analysis result path: %s.", analysis_result_dirpath)
    dct_name = "upliftMT"

    logger.debug("Writing to data file...")
    data.join([treatment_col, target_col]).to_csv(datatable_path, index=False)
    logger.debug("Done writing.")
    logger.debug("Building dictionary from data...")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            r"""^Sample datasets location does not exist \([^)]*/khiops_data/samples\)\.\s+"""
            r"""Execute the kh-download-datasets script or the khiops\.tools\.download_datasets function to download them\.$""",
            UserWarning, f"^{kh.internals.runner.__name__}$")
        kh.build_dictionary_from_data_table(
            datatable_filename, dct_name, str(dct_filepath))
    logger.debug("Done building dictionary.")
    logger.debug("Reading from dictionary file...")
    domain = kh.read_dictionary_file(str(dct_filepath))
    logger.debug("Done reading.")

    dct = domain.get_dictionary(dct_name)
    for var in categorical_vars:
        dct.get_variable(var).type = "Categorical"
    dct.get_variable(t).type = "Categorical"
    dct.get_variable(y).type = "Categorical"
    is_in_train_dataset_variable = kh.Variable()
    is_in_train_dataset_variable.name = "Y_T"
    is_in_train_dataset_variable.type = "Categorical"
    is_in_train_dataset_variable.used = True
    is_in_train_dataset_variable.rule = f"""Concat({y},"_",{t})"""
    dct.add_variable(is_in_train_dataset_variable)
    if outputdir is not None:
        logger.debug("Exporting dictionary to file...")
        domain.export_khiops_dictionary_file(str(dirpath / "dictionary.kdic"))
        logger.debug("Done exporting.")

    logger.debug("Training recoder...")
    analysis_result_files = kh.train_recoder(domain, dct_name, datatable_filename, "Y_T",
        str(analysis_result_dirpath / "predictor_analysis_result.khj"),
        sample_percentage=100, max_trees=0, max_pairs=0, max_parts=maxpartnumber or 0)
    logger.debug("Analysis result files: %s, %s.", analysis_result_files[0], analysis_result_files[1])
    logger.debug("Done training.")

    logger.debug("Reading analysis result file...")
    train_results = kh.read_analysis_results_file(analysis_result_files[0])
    logger.debug("Done reading.")

    model = {}
    groups_by_part_by_variable = {}
    levels = {}
    # Disable all input variables since we will create a filter variable for each one in turn
    # and it is that filter variable that will be enabled.
    for x in xs:
        dct.get_variable(x).used = False
    for x in xs:
        logger.info("(variable %r)  Computing uplift...", x)
        varmodel, groups_by_interval_for_variable, level = uplift_MODL_for_var(
            x, y, t, all_treatments, train_results, domain, dct, dct_name, datatable_filename, str(analysis_result_dirpath) + f"/{x}_%(interval)s_analysis_result.khj", maxtreatmentgroups)
        if varmodel is not None:
            model[x] = varmodel
        if groups_by_interval_for_variable is not None:
            groups_by_part_by_variable[x] = groups_by_interval_for_variable
        levels[x] = level
        logger.info("(variable %r)  Done computing uplift.", x)
        
    logger.info("Done computing uplift.")
        
    return model, groups_by_part_by_variable, sorted((var_level_pair for var_level_pair in levels.items()), key=lambda namelevel: (-namelevel[1], namelevel[0]))


def uplift_MODL_for_var(x, y, t, all_t_values, train_results, domain, dct, dct_name, datatable_filename, analysis_result_filename_template, maxtreatmentgroups):
    logger.debug("(variable %r)  Analysis result file name template: %s.", x, analysis_result_filename_template)
    pair_results = train_results.preparation_report.get_variable_statistics(x)
    level = pair_results.level
    logger.debug("(variable %r)  Level is %f.", x, level)

    if level == 0:
        return None, None, 0.0
    else:
        vardim = pair_results.data_grid.dimensions[0]
        logger.debug("(variable %r)  Partition type is %r.", x, vardim.partition_type)
        match vardim.partition_type:
            case "Value groups":
                partition = ValGrpPartition([ValGrp(valgrp.values) for valgrp in vardim.partition], next(i for i, valgrp in enumerate(vardim.partition) if valgrp.is_default_part))
            case "Intervals":
                partition = IntervalPartition([Interval(interval.lower_bound, interval.upper_bound) for interval in vardim.partition])
            case ptype: raise ValueError("unsupported partition type %r" % ptype)

    filtre_index_variable = kh.Variable()
    filtre_index_variable.name = "Filtre_%s" % x
    filtre_index_variable.type = "Categorical"
    filtre_index_variable.used = True
    filtre_index_variable.rule = str(partition_to_rule(partition, dct.get_variable(x)))
    logger.debug("(variable %r)  Filter index variable rule: %r.", x, filtre_index_variable.rule)
    dct.add_variable(filtre_index_variable)
    
    groups_by_part = {}
    for i, part in enumerate(partition):
        partname = f"I{i + 1}"
        logger.debug("(variable %r, part %s)  Training recoder...", x, part)
        analysis_result_files = kh.train_recoder(
            domain, dct_name, datatable_filename, y, analysis_result_filename_template % {"interval": partname},
            sample_percentage=100,
            selection_variable="Filtre_{}".format(x),
            selection_value=partname,
            max_trees=0,
            max_pairs=0,
            max_constructed_variables=0,
            max_text_features=0,
            max_parts=maxtreatmentgroups or 0)
        logger.debug("(variable %r, part %s)  Analysis result files: %s, %s.", x, part, analysis_result_files[0], analysis_result_files[1])
        logger.debug("(variable %r, part %s)  Done training.", x, part)
        
        logger.debug("(variable %r, part %s)  Reading analysis result file...", x, part)
        train_results = kh.read_analysis_results_file(analysis_result_files[0])
        logger.debug("(variable %r, part %s)  Done reading.", x, part)

        logger.debug("(variable %r, part %s)  Analysis result refers to these variable names: {%s}",
                     x, part, ", ".join(f"'{varname}'" for varname in train_results.preparation_report.get_variable_names()))

        if not train_results.preparation_report.target_values:
            logger.debug("(variable %r, part %s)  Empty preparation report.", x, part)
        group_results = train_results.preparation_report.get_variable_statistics(t)
        logger.debug("(variable %r, part %s)  Level of treatment %r is %f.", x, part, t, group_results.level)

        if group_results.level == 0:  # ==> Put all treatments into the same group.
            groups_by_part[part] = [tuple(map(str, all_t_values))]
        else:
            treatment_groups = group_results.data_grid.dimensions[0].partition
            logger.debug("(variable %r, part %s)  Groups before repairs: %s.", x, part, [grp.to_dict() for grp in treatment_groups])
            logger.debug("(variable %r, part %s)  Repairing groups...", x, part)
            groups_by_part[part] = repair_groups(treatment_groups, all_t_values)
            logger.debug("(variable %r, part %s)  Done repairing.", x, part)
            logger.debug("(variable %r, part %s)  Groups after repairs: %s.", x, part, groups_by_part[part])

    dct.remove_variable(filtre_index_variable.name)
        
    return partition, groups_by_part, level


def repair_groups(groups, all_treatments):
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


# class modele_E_y_avec_rapprochement_MODL(BaseEstimator, TransformerMixin):
#     def __init__(self, classifier):
#         self.classifier = classifier
    
#     def fit(self, X, nom_col_trait, nom_col_outcome, nom_col_X, groupements_par_intervalle, bornes_superieures):
#         self.classifiers_dict = {}
#         self.nom_col_trait = nom_col_trait
#         self.nom_col_outcome = nom_col_outcome
#         self.nom_col_X = nom_col_X
#         self.traitements = np.unique(X[self.nom_col_trait])
        
#         # Convertir les bornes en float pour le calcul
#         bornes_superieures = [float(b) for b in bornes_superieures]
        
#         # Créer une copie pour ne pas modifier le DataFrame original
#         X_combined = X.copy()
        
#         # Liste pour stocker les DataFrames dupliqués
#         dfs_a_ajouter = []
        
#         # Parcourir chaque intervalle et ses règles de regroupement
#         for i, (interval_name, groupes) in enumerate(groupements_par_intervalle.items()):
#             # Définir les bornes de l'intervalle courant
#             borne_inf = bornes_superieures[i-1] if i > 0 else -np.inf
#             borne_sup = bornes_superieures[i]
            
#             # Sélectionner les individus de cet intervalle
#             X_intervalle = X[(X[self.nom_col_X] >= borne_inf) & (X[self.nom_col_X] < borne_sup)]
            
#             # Parcourir les groupes de traitements à dupliquer
#             for groupe in groupes:
#                 # Gérer le cas du groupe avec 'tous les autres' (' * ')
#                 if ' * ' in groupe:
#                     t_specifique = [t for t in groupe if t != ' * '][0]
#                     traitements_autres = [t for t in self.traitements if t not in groupe]
                    
#                     # Dupliquer les individus du traitement spécifique vers les autres
#                     for t_cible in traitements_autres:
#                         df_copie = X_intervalle[X_intervalle[self.nom_col_trait] == t_specifique].copy()
#                         df_copie[self.nom_col_trait] = t_cible
#                         dfs_a_ajouter.append(df_copie)
                    
#                     # Dupliquer les individus des autres traitements vers le traitement spécifique
#                     df_autres_traitements = X_intervalle[X_intervalle[self.nom_col_trait].isin(traitements_autres)].copy()
#                     df_autres_traitements[self.nom_col_trait] = t_specifique
#                     dfs_a_ajouter.append(df_autres_traitements)
#                 else:
#                     # Gérer les groupes de taille > 1 sans ' * '
#                     for t_source in groupe:
#                         for t_cible in groupe:
#                             if t_source != t_cible:
#                                 df_copie = X_intervalle[X_intervalle[self.nom_col_trait] == t_source].copy()
#                                 df_copie[self.nom_col_trait] = t_cible
#                                 dfs_a_ajouter.append(df_copie)
        
#         # Concaténer le DataFrame original avec toutes les copies
#         if dfs_a_ajouter:
#             X_combined = pd.concat([X_combined] + dfs_a_ajouter, ignore_index=True)

#         # La partie restante de la méthode fit est inchangée, elle utilise maintenant X_combined
#         self.traitements = np.unique(X_combined[self.nom_col_trait])
#         for traitement in self.traitements:
#             X_trait = X_combined[X_combined[self.nom_col_trait] == traitement]
#             X_train_trait = X_trait.drop(columns=[self.nom_col_outcome, self.nom_col_trait])
#             y_train_trait = X_trait[self.nom_col_outcome]
#             if y_train_trait.nunique() > 1:
#                 classifier_clone = clone(self.classifier)
#                 classifier_clone.fit(X_train_trait, y_train_trait)
#                 self.classifiers_dict[traitement] = classifier_clone
#             else:
#                 single_class_predictor = SingleClassPredictor()
#                 single_class_predictor.fit(X_train_trait, y_train_trait)
#                 self.classifiers_dict[traitement] = single_class_predictor
#                 print(f"Avertissement : Le classifieur pour le traitement {traitement} a été remplacé par un prédicteur de classe unique.")
                
#         return self

#     def predict_e_y(self, X):
#         X_test=X.copy()
#         self.X_test_list = []
#         for value in self.traitements:
#             X_test_copy = X_test.copy()
#             X_test_copy[self.nom_col_trait] = value
#             X_test_copy = X_test_copy.drop(columns=[self.nom_col_trait])
#             self.X_test_list.append(1)   
#         probabilities_0 = self.classifiers_dict[self.traitements[0]].predict_proba(X_test_copy)[:, 1]
#         probabilities_1 = self.classifiers_dict[self.traitements[1]].predict_proba(X_test_copy)[:, 1]
#         uplift_values=[]
#         for i in range(len(probabilities_0)):
#             uplift_values.append([probabilities_0[i]])
#             uplift_values.append([probabilities_1[i]])
#         i = 2
#         all_probabilities = []
#         for i in range(len(self.classifiers_dict)):
#             probabilities = self.classifiers_dict[self.traitements[i]].predict_proba(X_test_copy)[:, 1]
#             all_probabilities.append(probabilities)
        
#         return all_probabilities

# class SingleClassPredictor(BaseEstimator, ClassifierMixin):
#     def fit(self, X, y=None):
#         self.unique_class_ = np.unique(y)[0]
#         self.classes_ = np.array([0, 1])  # Explicitly state classes for consistency
#         return self

#     def predict(self, X):
#         return np.full(X.shape[0], self.unique_class_)

#     def predict_proba(self, X):
#         probabilities = np.zeros((X.shape[0], 2)) 
        
#         # Determine the index of the unique class (0 or 1)
#         class_index = int(self.unique_class_) 
        
#         # Set the probabilities for the unique class to 1.0
#         probabilities[:, class_index] = 1.0
        
#         return probabilities


# def calculer_probas_groupees(df, groupements_par_intervalle, bornes_superieures_str):
    
#     global_mean_y = df['Y'].mean()

#     # Cas spécial : aucun groupement ni intervalle fourni
#     if not groupements_par_intervalle and not bornes_superieures_str:
#         resultats_plats = []
#         interval_label = "[-inf, +inf)" # Un seul intervalle global
#         traitements_uniques = sorted(df['T'].unique())
#         # Calculer la proba pour chaque traitement séparément
#         for t in traitements_uniques:
#             df_traitement = df[df['T'] == t]
#             if df_traitement.empty:
#                 probabilite = np.nan
#             else:
#                 probabilite = df_traitement['Y'].mean()
            
#             resultats_plats.append({
#                 'Intervalle': interval_label,
#                 'Traitement': t,
#                 'Probabilite_Y1': probabilite
#             })
        
#         df_resultat_long = pd.DataFrame(resultats_plats)
        
#         # Pivoter
#         df_pivot = df_resultat_long.pivot_table(
#             index='Traitement', 
#             columns='Intervalle', 
#             values='Probabilite_Y1'
#         )
        
#         # Reindexer pour inclure tous les traitements attendus
#         traitements_attendus = range(df['T'].min(), df['T'].max() + 1)
#         df_pivot = df_pivot.reindex(traitements_attendus)
#         df_pivot = df_pivot.fillna(global_mean_y)
#         return df_pivot
    
#     # --- Logique originale ---
    
#     bornes_superieures = [float(b) for b in bornes_superieures_str]
    
#     resultats_plats = []
#     borne_inf_filter = -np.inf 
    
#     for i, borne_sup in enumerate(bornes_superieures):
#         interval_name = f"I{i+1}" 
        
#         if interval_name not in groupements_par_intervalle:
#             borne_inf_filter = borne_sup 
#             continue
            
#         borne_inf_label = 0.0 if i == 0 else bornes_superieures[i-1]
#         interval_label = f"[{borne_inf_label}, {borne_sup})" 
            
#         df_intervalle = df[(df['X'] >= borne_inf_filter) & (df['X'] < borne_sup)]
        
#         groupes_str_list = groupements_par_intervalle[interval_name]
#         for groupe_str in groupes_str_list:
            
#             groupe_int = [int(float(t)) for t in groupe_str]
            
#             df_groupe = df_intervalle[df_intervalle['T'].isin(groupe_int)]
            
#             if df_groupe.empty:
#                 probabilite = np.nan
#             else:
#                 probabilite = df_groupe['Y'].mean()
            
#             for t_str in groupe_str:
#                 resultats_plats.append({
#                     'Intervalle': interval_label,
#                     'Traitement': int(t_str), 
#                     'Probabilite_Y1': probabilite
#                 })

#         borne_inf_filter = borne_sup 
        
#     df_resultat_long = pd.DataFrame(resultats_plats)
    
#     df_resultat_long = df_resultat_long.drop_duplicates(subset=['Intervalle', 'Traitement'])
    
#     df_pivot = df_resultat_long.pivot_table(
#         index='Traitement', 
#         columns='Intervalle', 
#         values='Probabilite_Y1'
#     )
    
#     traitements_attendus = sorted(df['T'].unique())
#     df_pivot = df_pivot.reindex(traitements_attendus)
#     df_pivot = df_pivot.fillna(global_mean_y)
#     return df_pivot

# def predict_probas_from_pivot(X_test, df_probas_pivot):
#     labels = df_probas_pivot.columns
    
#     try:
#         bins = [float(labels[0].split(', ')[0].strip('['))]
#         bins.extend([float(col.split(', ')[1].strip(')')) for col in labels])
#         bins[0] = -np.inf
#         bins[-1] = np.inf
#     except Exception as e:
#         raise e

#     X_test_copy = X_test.copy()
    
#     X_test_copy['Intervalle'] = pd.cut(
#         X_test_copy['X'], 
#         bins=bins, 
#         labels=labels, 
#         right=False, 
#         include_lowest=True
#     )
    
#     df_lookup = df_probas_pivot.T.reset_index()
    
#     X_test_with_probas = X_test_copy.merge(
#         df_lookup, 
#         on='Intervalle', 
#         how='left'
#     )
    
#     traitement_cols = sorted(df_probas_pivot.index)
#     output_list_of_arrays = []
    
#     for t in traitement_cols:
#         arr = X_test_with_probas[t].values.astype(np.float32)
#         output_list_of_arrays.append(arr)
        
#     return output_list_of_arrays

# def calculate_rmse(true_values, predictions):
#     return np.sqrt(np.mean((true_values - predictions) ** 2))

# def critere_modl(df, col_X, col_T, col_Y, intervalles, regroupements_traitements):
#     """
#     Calcule la valeur de l'expression mathématique complète.

#     Args:
#         df (pd.DataFrame): Le dataset.
#         col_X (str): Le nom de la colonne X (variable continue).
#         col_T (str): Le nom de la colonne T (traitements).
#         col_Y (str): Le nom de la colonne Y (variable cible).
#         intervalles (list): Liste des intervalles, e.g., [[0, 0.1], [0.1, 0.2]].
#         regroupements_traitements (list): Liste des regroupements de traitements par intervalle. Exemple : [[[0,2],[3,1]], [[0,3],[2,1]]]

#     Returns:
#         float: Le résultat de l'expression.
#     """
    
#     N = len(df)
#     I = len(intervalles)
#     T = df[col_T].nunique()
#     J = df[col_Y].nunique()
    
#     total_somme = 0.0

#     # Terme 1: log(N) OK
#     total_somme += np.log(N)

#     # Terme 2: log(C(N+I-1, I-1)) OK
#     # Utilisation de gammaln(n+1) = log(n!)
#     total_somme += gammaln(N + I) - gammaln(I) - gammaln(N + 1)

#     # Terme 3: I * log(T) OK
#     #total_somme += 20 * I * np.log(T)
#     total_somme += I * np.log(T)

#     # Terme 4: Somme des log(Beta(T, G_i)) OK
#     # G_i est le nombre de groupes dans l'intervalle i.

#     for i in range(I):
#         if i < len(regroupements_traitements):
#             G_i = len(regroupements_traitements[i])
#             total_somme += KWStat.LnBell(T, G_i)

#     # Terme 5: Double sommation OK
#     for i in range(I):
#         if i < len(regroupements_traitements):
#             # Obtention des groupes de traitements pour l'intervalle i
#             groupes_traitements_0 = regroupements_traitements[i]
#             groupes_traitements = [[int(element) for element in sous_liste] for sous_liste in groupes_traitements_0]
#             # Filtrage des données pour l'intervalle i
#             df_i = df[(df[col_X] >= intervalles[i][0]) & (df[col_X] < intervalles[i][1])].copy()
#             for groupe_g in groupes_traitements:
#                 # Filtrage des données pour le groupe de traitements g
#                 df_ig = df_i[df_i[col_T].isin(groupe_g)].copy()
#                 N_ig = len(df_ig)
                
#                 # Terme A: log(C(N_ig + J - 1, J - 1))
#                 terme_A = gammaln(N_ig + J) - gammaln(J) - gammaln(N_ig + 1)
                
#                 # Terme B: log(N_ig!)
#                 terme_B = gammaln(N_ig + 1)
                
#                 # Terme C: Somme des -log(N_igj!)
#                 terme_C = 0.0
#                 for y_val in df_ig[col_Y].unique():
#                     N_igj = len(df_ig[df_ig[col_Y] == y_val])
#                     terme_C += -gammaln(N_igj + 1)

#                 total_somme += terme_A + terme_B + terme_C

#     return total_somme

# def dict_to_ordered_list(data_dict):
#   sorted_items = sorted(data_dict.items(), key=lambda item: int(item[0][1:]))
  
#   ordered_list = [value for key, value in sorted_items]
  
#   return ordered_list

# def applique_MODL(X, nom_col_trait, nom_col_outcome, nom_col_X, groupements_par_intervalle, bornes_superieures):
#     bornes_superieures = [float(b) for b in bornes_superieures]
    
#     X_combined = X.copy()
#     dfs_a_ajouter = []
    
#     for i, (interval_name, groupes) in enumerate(groupements_par_intervalle.items()):
#         borne_inf = bornes_superieures[i-1] if i > 0 else -np.inf
#         borne_sup = bornes_superieures[i]
        
#         X_intervalle = X[(X[nom_col_X] >= borne_inf) & (X[nom_col_X] < borne_sup)]
        
#         for groupe in groupes:
#             for t_source in groupe:
#                 for t_cible in groupe:
#                     if t_source != t_cible:
#                         df_copie = X_intervalle[X_intervalle[nom_col_trait] == int(t_source)].copy()
#                         df_copie[nom_col_trait] = t_cible
#                         dfs_a_ajouter.append(df_copie)
#     if dfs_a_ajouter:
#         X_combined = pd.concat([X_combined] + dfs_a_ajouter, ignore_index=True)
    
#     return X_combined

# class S_Learner(BaseEstimator, TransformerMixin):
#     def __init__(self, classifier):
#         self.classifier = classifier
        
#     def fit(self, X_train, y_train,nom_col_trait):
#         self.classifier.fit(X_train, y_train)
#         self.nom_col_trait=nom_col_trait
#         self.traitements = np.unique(X_train[nom_col_trait])
#         return self
        
#     def predict_uplift(self, X):
#         X_test=X.copy()
#         self.X_test_list = []
#         for value in self.traitements:
#             X_test_copy = X_test.copy()
#             X_test_copy[self.nom_col_trait] = value
#             self.X_test_list.append(X_test_copy)

#         probabilities_0 = self.classifier.predict_proba(self.X_test_list[0])[:, 1]
#         probabilities_1 = self.classifier.predict_proba(self.X_test_list[1])[:, 1]
#         uplift_values=[]
#         for i in range(len(probabilities_0)):
#             uplift_values.append([probabilities_1[i] - probabilities_0[i]])
#         for X_test in self.X_test_list[2:]:
#             probabilities = self.classifier.predict_proba(X_test)[:, 1]
#             uplift = probabilities - probabilities_0
#             for i in range(len(uplift_values)):
#                 l=[]
#                 ll=[]
#                 for j in uplift_values[i]:
#                     ll.append(j)
#                 ll.append(uplift[i])
#                 uplift_values[i] = ll
#         #return np.array(self.X_test_list)
#         uplift_values = np.array(uplift_values)
#         return uplift_values
    
#     def predict_policy(self, X):
#         uplift_values = self.predict_uplift(X)
#         indices = np.argmax(uplift_values, axis=1) + 1
#         for i, values in enumerate(uplift_values):
#             if np.all(values <= 0):
#                 indices[i] = 0 
#         return indices

#     def predict_worst_policy(self, X):
#         uplift_values = self.predict_uplift(X)
#         indices = np.argmin(uplift_values, axis=1) + 1
#         for i, values in enumerate(uplift_values):
#             if np.all(values >= 0):
#                 indices[i] = 0 
#         return indices
