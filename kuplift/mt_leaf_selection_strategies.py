# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from __future__ import annotations

from math import exp
from typing import Any
import numpy as np


VALID_LEAF_SELECTION_STRATEGIES = {
    "best_leaf",
    "random",
    "weighted_random",
}


def validate_leaf_selection_strategy(strategy: str) -> None:
    if strategy not in VALID_LEAF_SELECTION_STRATEGIES:
        raise ValueError(
            f"Unknown leaf selection strategy {strategy!r}. "
            f"Expected one of {sorted(VALID_LEAF_SELECTION_STRATEGIES)}."
        )


def select_leaf(
    strategy: str,
    node_vs_cost: dict[Any, float],
    rng: np.random.Generator,
    temperature: float = 1.0,
):
    """
    Select a node according to the requested strategy.

    Parameters
    ----------
    strategy : str
        One of: "best_leaf", "random", "weighted_random".
    node_vs_cost : dict
        Mapping node -> cost (lower is better).
    rng : np.random.Generator
        Random generator used for stochastic strategies.
    temperature : float, default=1.0
        Softmax-like temperature used by weighted_random.
        Lower => more greedy toward best cost.

    Returns
    -------
    node
        The selected node.

    Raises
    ------
    ValueError
        If strategy is unknown or node_vs_cost is empty.
    """
    validate_leaf_selection_strategy(strategy)

    if not node_vs_cost:
        raise ValueError("node_vs_cost must not be empty")

    nodes = list(node_vs_cost.keys())

    if strategy == "best_leaf":
        return min(node_vs_cost, key=node_vs_cost.get)

    if strategy == "random":
        idx = rng.integers(0, len(nodes))
        return nodes[idx]

    # weighted_random
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    costs = np.array([node_vs_cost[n] for n in nodes], dtype=float)
    cmin = float(np.min(costs))

    # Convert "lower is better" costs to positive weights
    # w_i = exp(-(c_i - cmin)/temperature)
    # TODO: Use levels instead of this computation. See TODO.md.
    weights = np.array([exp(-(c - cmin) / temperature) for c in costs], dtype=float)
    wsum = float(np.sum(weights))

    # Fallback safety if numerical underflow occurs
    if wsum <= 0 or not np.isfinite(wsum):
        idx = int(np.argmin(costs))
        return nodes[idx]

    probs = weights / wsum
    idx = int(rng.choice(len(nodes), p=probs))
    return nodes[idx]
