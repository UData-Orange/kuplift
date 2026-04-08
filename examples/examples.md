Examples + tests
================

These examples are here for documentation and testing purposes.
They may be executed using Python's built-in module `doctest`.

Imports
-------

~~~ pycon

>>> from khiops import core as kh
>>> from kuplift import *

~~~


Convert a partition to a Khiops Rule
------------------------------------

~~~ pycon

>>> var = kh.Variable()
>>> var.name = "VAR1"
>>> partition_to_rule(IntervalPartition([Interval(1.2, 3.4), Interval(3.4, 5.6), Interval(5.6, 7.8)]), var)
IntervalId(IntervalBounds(3.4, 5.6), VAR1)
>>> partition_to_rule(ValGrpPartition([ValGrp(["a", "b"]), ValGrp(["c", "d", "e"]), ValGrp(["f"])], 1), var)
GroupId(ValueGroups(ValueGroup("a", "b"), ValueGroup("c", "d", "e", " * "), ValueGroup("f")), VAR1)

~~~