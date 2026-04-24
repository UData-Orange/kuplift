import typing
import khiops.core

VarType = typing.Literal["Numerical", "Categorical"]
Part = khiops.core.PartInterval | khiops.core.PartValue | khiops.core.PartValueGroup
Partition = list[khiops.core.PartInterval] | list[khiops.core.PartValue] | list[khiops.core.PartValueGroup]
NumVarPartition = list[khiops.core.PartInterval]
CatVarPartition = list[khiops.core.PartValueGroup]