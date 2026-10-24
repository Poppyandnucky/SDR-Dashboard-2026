"""Stage 6: fixed-point updates for shared facility mortality parameters."""

from __future__ import annotations


def update(facility_baseline, l23_rr, modeled_l4, modeled_l23, target_l4, target_l23):
    """New (p_mat_death_facility_baseline, RR_MM_l23_vs_l4), shared by the counties.

    facility_baseline  current p_mat_death_facility_baseline (the value simulated with)
    l23_rr             current RR_MM_l23_vs_l4: L2/3 death risk relative to L4
    modeled_l4         expected deaths per 100,000 deliveries at final L4 (sum of
                       individual death probabilities, p_mat_death), pooled over
                       the counties and runs
    modeled_l23        the same at final L2/3
    target_l4          KHIS internal-L4 maternal deaths per 100,000 deliveries,
                       pooled over the counties, x 0.618 (direct, non-abortion
                       share): 30.8 (calibration_common.khis_mmr)
    target_l23         the same at L2/3: 7.1

    Expected L4 deaths are proportional to the baseline, and L2/3 deaths to the
    baseline x RR, so A = target_l4 / modeled_l4 rescales the baseline and the
    rest of the L2/3 gap goes to the RR.
    """
    l4_adjustment = target_l4 / modeled_l4
    return (
        facility_baseline * l4_adjustment,
        l23_rr * (target_l23 / modeled_l23) / l4_adjustment,
    )

