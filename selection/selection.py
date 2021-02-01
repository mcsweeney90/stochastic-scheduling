#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of stochastic HEFT extensions.
Ground rules: 1. No spoilation.
              2. MC method used to evaluate schedule makespan.
"""


# First, just use the standard HEFT upward ranking.

# =============================================================================
# Different priorities
# =============================================================================

"""
Options.
1. Standard HEFT scalar mean values.
2. HEFT-WM.
Multiple different stochastic upward ranks (standard averaged graph):
3. Sculli's method.
4. CorLCA.
5. MC with different numbers of samples and distributions.
Weighted average graph. Different ways to weight?
"""


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