from dataclasses import dataclass
import khiops.core
from .typealiases import CatVarPartition, Part
from .preparation_report import VarStats, Stats

TreatmentGroup = khiops.core.PartValueGroup
TreatmentGroups = CatVarPartition

@dataclass(frozen=True)
class VarStatsWithGroups(VarStats):
    """Statistics for a variable, enriched with treatment-grouping information."""
    groups_by_parts: dict[Part, TreatmentGroups]
    groups_by_treatments_by_parts: dict[Part, dict[str, TreatmentGroup]]


ModelWithGroups = dict[str, VarStatsWithGroups]


@dataclass(frozen=True)
class StatsWithGroups(Stats):
    """Statistics for all variables, enriched with treatment-grouping information."""
    model: ModelWithGroups


def get_treatment_groups_of_var(model: ModelWithGroups, variable: str) -> dict[Part, TreatmentGroups]:
    return model[variable].groups_by_parts


def get_treatment_groups(model: ModelWithGroups) -> dict[str, dict[Part, TreatmentGroups]]:
    return {varname: varstats.groups_by_parts for varname, varstats in model.items()}