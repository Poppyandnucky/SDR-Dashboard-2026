"""Stage 1: preterm birth (PT_scale).

PT_scale multiplies the probability of each gestational week below 37
(global_func.GA_assign_kenya), so the overall preterm rate is proportional to it.
The target is the preterm share of all facility births after transfer, which also
depends on routing and transfers, so the update is repeated each round from the
previous round's simulation and damped (a full step can overshoot):

    PT_scale <- PT_scale x (target / modeled)^damping
"""

from __future__ import annotations


def update(pt_scale, target, modeled, damping=0.5):
    """New PT_scale for one county.

    pt_scale  current PT_scale (the value the round was simulated with)
    target    preterm share of all facility births after transfer: 0.134
              (Waiswa et al. 2020; calibration_data.PRETERM_FAC)
    modeled   the same share in the simulation: women delivering at L2/3, L4,
              or L5 (after any transfer) who are preterm, pooled over the round's
              runs (calibration_common.summarize, "preterm_fac")
    damping   exponent on the correction, between 0 and 1: 1 takes the full
              proportional step, 0.5 (default) takes the square root of it,
              e.g. modeled 0.10 -> x 1.34 at 1, x 1.16 at 0.5
    """
    return pt_scale * (target / modeled) ** damping
