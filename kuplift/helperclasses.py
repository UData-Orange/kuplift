from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Optional
import math


class Partition(ABC):
    @property
    @abstractmethod
    def parts(self):
        pass

    def __iter__(self):
        return iter(self.parts)

    def __eq__(self, other):
        return self.parts == other.parts
    

class Part:
    pass
    

@dataclass(frozen=True)
class ValGrp(Part):
    values: list

    def __repr__(self):
        return f"ValGrp({self.values!r})"

    def __str__(self):
        return "{%s}" % ", ".join(self.values)
    
    def __contains__(self, x):
        return x in self.values
    
    def __hash__(self):
        return hash(tuple(self.values))


class ValGrpPartition(Partition):
    """Partition of type 'value groups'.
    
    Attributes
    ----------
        groups: Sequence[ValGrp]
            The groups. Each group is an iterable of its values.
        defaultgroupindex: int
            The group index affected to transformed elements when they do not explicitly appear in any group.
    """

    def __init__(self, groups: Sequence[ValGrp], defaultgroupindex: int):
        if not groups:
            raise ValueError("there must be at least one group")
        if defaultgroupindex < 0 or defaultgroupindex >= len(groups):
            raise ValueError(f"default group index is {defaultgroupindex} but groups are numbered from 0 to {len(groups) - 1}")
        self.groups = groups
        self.defaultgroupindex = defaultgroupindex

    @property
    def parts(self):
        return self.groups

    def transform(self, col):
        return col.transform(self.transform_elem)

    def transform_elem(self, elem):
        for i, group in enumerate(self.groups):
            if str(elem) in group:
                return i
        return self.defaultgroupindex
    
    def __repr__(self):
        return f"ValGrpPartition({self.groups!r}, {self.defaultgroupindex!r})"

    def __str__(self):
        return """
Value group partition
    {ngroups} groups ("*" indicates the default group):
{groups}
"""[1:-1].format(
    ngroups=len(self.groups),
    groups="\n".join(f"      {'*' if i == self.defaultgroupindex else ' '} - {group}" for i, group in enumerate(self.groups))
)


@dataclass(frozen=True)
class Interval(Part):
    lower: Optional[float] = None
    upper: Optional[float] = None
        
    @property
    def catches_missing(self):
        return self.lower is None or self.upper is None
    
    def __repr__(self):
        return "Interval({}, {})".format(self.lower, self.upper)
    
    def __str__(self):
        return "[]" if self.catches_missing else f"]{self.lower}, {self.upper}]"
    
    def __contains__(self, x):
        return not self.catches_missing and (self.lower < x <= self.upper)
    
    def __bool__(self):
        return not self.catches_missing


class IntervalPartition(Partition):
    """Partition of type 'intervals'.
    
    Attributes
    ----------
        intervals: Sequence[Interval]
            The intervals. Each interval is a pair defining its lower bound and its upper bound (in that order).
            The exception to this rule is the empty interval representing 'MISSING' values. If present, it must be
            the first interval of the sequence.
    """

    def __init__(self, intervals: Sequence[Interval]):
        if not intervals:
            raise ValueError("there must be at least one interval")
        if intervals[0].catches_missing:
            if len(intervals) == 1:
                raise ValueError("there must be at least one \"non-MISSING\" interval")
            intervals[1] = Interval(-math.inf, intervals[1].upper)
        else:
            intervals[0] = Interval(-math.inf, intervals[0].upper)
        intervals[-1] = Interval(intervals[-1].lower, +math.inf)
        self.intervals = intervals

    @property
    def parts(self):
        return self.intervals

    def transform(self, col):
        return col.transform(self.transform_elem)

    def transform_elem(self, elem):
        if not isinstance(elem, (int, float)) or math.isnan(elem):
            return 0
        for i, interval in enumerate(self.intervals):
            if elem in interval:
                return i
            
    def __repr__(self):
        return f"IntervalPartition({self.intervals!r})"
    
    def __str__(self):
        return """
Interval partition
    {nintervals} intervals:
{intervals}
"""[1:-1].format(
    nintervals=len(self.intervals),
    intervals="\n".join(f"      - {interval}" for interval in self.intervals)
)
    

@dataclass
class TargetTreatmentPair:
    """Target-treatment pair.

    Used to identify both a target and a treatment.
    This class only exists for the purpose of formatting.
    """

    target: object
    treatment: object

    def __hash__(self):
        return hash((self.target, self.treatment))
    
    def __str__(self):
        return "(%s|%s)" % (self.target, self.treatment)
    

@dataclass
class TargetTreatmentGroupPair:
    """Target-treatmentgroup pair.
    
    Used to identify both a target and a treatment group.
    This class only exists for the purpose of formatting.
    """

    target: object
    treatment_group: tuple[object]

    def __hash__(self):
        return hash((self.target, self.treatment_group))
    
    def __str__(self):
        return "(%s|%s)" % (self.target, ",".join(self.treatment_group))