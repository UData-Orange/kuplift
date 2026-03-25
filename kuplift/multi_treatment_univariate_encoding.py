# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

"""Multi-treatment Univariate Encoding

This module contains everything needed to make univariate variable transformation
capable of merging treatments that give similar outcome.

The main class of this module is 'MultiTreatmentUnivariateEncoding'.
"""


from .helperclasses import ValGrp, ValGrpPartition, Interval, IntervalPartition, TargetTreatmentPair


class MultiTreatmentUnivariateEncoding:
    """
    The MultiTreatmentUnivariateEncoding class makes use of the khiops Python wrapper
    and enables one to fit and transform data while grouping treatments giving similar
    outcome.
    """

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
    

    def fit(self, data, treatment_col, target_col, maxpartnumber = None):
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
        """
        raise NotImplementedError


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
        raise NotImplementedError

    
    def fit_transform(self, data, treatment_col, target_col, maxpartnumber = None):
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

        Returns
        -------
        pd.Dataframe
            Pandas Dataframe that contains encoded data.
        """
        self.fit(data, treatment_col, target_col, maxpartnumber)
        return self.transform(data)
    

    def get_levels(self):
        """Get the level of all variables.
        
        Returns
        -------
        list[tuple[str, float]]
            (variable-name, variable-level) pairs in decreasing level order.
        """
        raise NotImplementedError
    

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
        raise NotImplementedError
    

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
        raise NotImplementedError
    

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
        raise NotImplementedError
    

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
        raise NotImplementedError
    

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
        raise NotImplementedError