# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from __future__ import annotations

import math


class KWStat:
    """
    Minimal KWStat subset used by multi-treatment DecisionTree cost models.
    """

    _ln_factorial_cache = [0.0]  # ln(0!) = 0

    @staticmethod
    def LnFactorial(n: int) -> float:
        """
        ln(n!)
        """
        if n < 0:
            raise ValueError("n must be >= 0")
        cache = KWStat._ln_factorial_cache
        current_size = len(cache)
        if n >= current_size:
            for i in range(current_size, n + 1):
                cache.append(cache[-1] + math.log(i))
        return cache[n]

    @staticmethod
    def LnStar(n: int) -> float:
        """
        Rissanen log-star in base 2 (sum of iterated positive logs).
        """
        if n <= 0:
            raise ValueError("n must be > 0")
        d_log2 = math.log(2.0)
        cost = 0.0
        x = math.log(float(n)) / d_log2
        while x > 0:
            cost += x
            x = math.log(x) / d_log2
        return cost

    @staticmethod
    def NaturalNumbersUniversalCodeLength(n: int) -> float:
        """
        Rissanen universal code length for natural integers n>=1.
        Returned in natural-log scale for consistency with tree costs.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        dC0 = 2.86511
        d_log2 = math.log(2.0)
        # ln-scale:
        return (math.log(dC0) / d_log2 + KWStat.LnStar(n)) * d_log2

    @staticmethod
    def LnBellNumber(n: int, k: int) -> float:
        """
        ln(Bell(n, k)) where Bell(n,k) is the number of partitions
        of n labeled items into at most k non-empty groups.

        Computed via Stirling numbers of the second kind:
          Bell(n,k) = sum_{i=1..k} S(n,i)
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        if k < 1:
            raise ValueError("k must be >= 1")
        k = min(k, n)

        # DP for S(n, i)
        # S(0,0)=1 ; S(a,0)=0 for a>0
        prev = [0] * (k + 1)
        prev[0] = 1

        for a in range(1, n + 1):
            curr = [0] * (k + 1)
            upper_i = min(a, k)
            for i in range(1, upper_i + 1):
                # S(a,i) = S(a-1,i-1) + i*S(a-1,i)
                curr[i] = prev[i - 1] + i * prev[i]
            prev = curr

        bell_n_k = sum(prev[1 : k + 1])
        return math.log(float(bell_n_k))
