# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from kuplift.optimized_univariate_encoding import OptimizedUnivariateEncoding
from kuplift.multi_treatment_univariate_encoding import MultiTreatmentUnivariateEncoding


@dataclass(frozen=True)
class EncodingSelectionResultV3:
    encoder_name: str
    treatment_modality_count: int


def count_treatment_modalities(treatment_col: pd.Series) -> int:
    if treatment_col is None:
        raise ValueError("treatment_col must not be None")
    return int(treatment_col.nunique(dropna=False))


def select_univariate_encoder_v3(treatment_col: pd.Series):
    """
    Select univariate encoder according to treatment modality count.

    Rules
    -----
    - 2 modalities  -> OptimizedUnivariateEncoding (OUE)
    - 3+ modalities -> MultiTreatmentUnivariateEncoding (MTUE)
    """
    n_modalities = count_treatment_modalities(treatment_col)

    if n_modalities < 2:
        raise ValueError(
            f"At least 2 treatment modalities are required, got {n_modalities}."
        )

    if n_modalities == 2:
        encoder = OptimizedUnivariateEncoding()
        info = EncodingSelectionResultV3(
            encoder_name="OUE",
            treatment_modality_count=n_modalities,
        )
    else:
        encoder = MultiTreatmentUnivariateEncoding()
        info = EncodingSelectionResultV3(
            encoder_name="MTUE",
            treatment_modality_count=n_modalities,
        )

    return encoder, info
