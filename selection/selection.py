#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of stochastic HEFT extensions.
Ground rules: 1. No spoilation.
              2. MC method used to evaluate schedule makespan.
"""


# First, just use the standard HEFT upward ranking.


# =============================================================================
# Priority functions.
# =============================================================================

"""
1. If scalar, just identity.
2. UCB.
3. Mean only.
4. 99th percentile.
"""

# =============================================================================
# Processor selection.
# =============================================================================

"""
Two steps to this: how to compute the (RV) estimates (e.g., lookahead, add future estimate). Then the multi-objective selection.
Scalarization is the only real alternative here, some choices being:
1. Mean only.
2. UCB.
3. Angle-based.
"""