"""Stages 1 to 5 solved together for the calibrated counties.

The targets of these stages are measured after emergency transfer, so they are
solved by repeating explicit steps until nothing changes. Each round runs the
stages in order; "modeled" means the value in the round's simulations.

Stage 1  Preterm birth (stages/stage1_preterm.py), damped proportional update from
         the previous round's simulation (round 1: the workbook value):
             PT_scale <- PT_scale x (0.134 / modeled P(preterm | facility birth))^0.5

Stage 2  Routing (route_county, stages/stage2_routing.py), closed form. With a the
         DHS 4+ ANC share, F_H|noANC and F_H|ANC the DHS home shares, q the KHIS
         2022 L2/3 share of facility births, theta = OR_recognized_L45 = 2, tau the
         transfer share from Stage 3, and r the recognition rate:
             r     = 1 - [1 - p_highrisk sen_risk_trad - (1 - p_highrisk)(1 - spec_risk_trad)] (1 - g)
                     (g = P(dated preterm or post-term | 4+ ANC), which depends on PT_scale)
             home_noANC   h_0 = F_H|noANC
             home_lowrisk h_U = F_H|ANC / (1 - r)
             home share     H = (1 - a) h_0 + a F_H|ANC
             initial L2/3   I = q (1 - H) / (1 - tau)
             baseline_l23_preference b is the root of
                 (1 - a)(1 - h_0) b + a [(1 - r)(1 - h_U) b + r b / (b + theta (1 - b))] = I
             p_lb_l23_45 = tau I;  p_ANC_base = a;
             base_LB = total births x (H, q (1 - H), L4/5 rest split by the KHIS L5 share)
         Repeated until base_LB and p_ANC_base stop changing (they enter r).

Stage 3  Transfers from L2/3 (fit_transfers, stages/stage3_transfers.py), counting
         equations re-simulated until stable. One simulation splits women who start
         at L2/3 and are not moved by the predicted or severe pathways into
             A  preterm, no complication: E_pt women
             B  non-severe complication:  E_ns women, preterm share pi_B, e_B events each
             C  all others:               N_C women, PT_C preterm, ev_C events
         With p_pt = t_l23_l45_preterm, p_ns = t_l23_l45_notsevere, and s4 the share
         of these transfers received at L4 (N_L4, ev_L4: other final-L4 births, events):
             P(preterm | L2/3) = [PT_C + E_pt (1 - p_pt) + E_ns (1 - p_ns) pi_B]
                                 / [N_C + E_pt (1 - p_pt) + E_ns (1 - p_ns)]
             R_L23:L4 = ([ev_C + E_ns (1 - p_ns) e_B] / N_L23)
                        / ([ev_L4 + s4 E_ns p_ns e_B] / [N_L4 + s4 (E_pt p_pt + E_ns p_ns)])
         are solved for (p_pt, p_ns) against the KHIS L2/3 preterm share and the
         L2/3:L4 complication ratio. t_l23_l45_predicted is rescaled so that, with
         w = 248 / 425 = 0.583 the Rescue labour-emergency share of known-reason transfers,
             predicted transfers = w / (1 - w) x (severe + preterm + non-severe transfers).
         tau <- realized share of L2/3 starters transferred.

Stage 4  Planned CS (update_stages_4_5, stages/stage4_planned_cs.py), closed form.
         q_H,c = p_elec_CS|highrisk (county c), q_P = p_elec_CS|preterm (shared);
         s_H,c and s_P,c are the shares of the two eligible groups (ANC, starting at
         L4/5, predicted high risk or predicted preterm) among final-L4 births, t_H,c
         and t_P,c their shares among preterm women starting at L4/5, w_c the county's
         share of those preterm women (a, b, c, d and weights in the code):
             s_H,c q_H,c + s_P,c q_P = 0.40 x KHIS L4 CS rate_c          (each county)
             sum_c w_c (t_H,c q_H,c + t_P,c q_P) = 0.265 x 1232 / 4477    (pooled)
         Substituting q_H,c from the first equations leaves one linear equation in q_P.

Stage 5  Complications and CS capacity (update_stages_4_5, stages/stage5_complications.py),
         proportional updates:
             x <- x target_x / modeled_x   for p_aph, p_ruptured_uterus, p_eclampsia,
                   p_OL_scale, pph_incidence_multiplier, sepsis_incidence_multiplier
                   (targets: internal-L4 rates, county total x pooled cause mix)
             p_cs_capacity[L4]   <- p_cs_capacity[L4] x (cs_l4 target - planned CS - transferred-in
                                    emergency CS) / emergency CS among L4/5 starters
                                    (all as shares of final-L4 births)
             p_cs_capacity[L2/3] <- p_cs_capacity[L2/3] x cs_l23 target / modeled L2/3 CS

Stages 4 and 5 are computed from the final Stage 3 simulation of the round.

Each run uses its own parameter draw (seed 42 + run), as the scenario runs do, so
targets are matched on average over parameter uncertainty; simulation seeds are
fixed. Outputs (--output-dir): values.csv, round_<n>.csv, final.csv.

Usage:
    python Calibration/calibrate_stages_1_to_5.py --output-dir Calibration/results/<run>
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Calibration.calibration_common import (
    CALIBRATED_COUNTIES, DEFAULTS, apply_values, county_targets, observed_locations, parameter_draw, save_values,
    simulate, summarize,
)
from Calibration.calibration_data import DHS, RESCUE
from Calibration.stages import stage1_preterm as preterm, stage2_routing as routing, stage3_transfers as transfers
from Calibration.stages import stage4_planned_cs as planned_cs, stage5_complications as complications
from parameter_loader import calculate_derived_parameters


# Transfer share of L2/3 starters before the first Stage 3 result.
STARTING_TAU = 0.08
# Stage 2 values set by route_county (with p_ANC_base and base_LB).
ROUTING_PARAMETERS = ("home_noANC", "home_lowrisk", "baseline_l23_preference", "p_lb_l23_45")
CS_CAPACITY = {"p_cs_capacity_l23": complications.CS_CAPACITY_L23, "p_cs_capacity_l4": complications.CS_CAPACITY_L4}
STAGE_5 = (*((name, None) for name in complications.INCIDENCE), *(("p_cs_capacity", i) for i in CS_CAPACITY.values()))
# Model shares reported each round without a target.
DIAGNOSTICS = ("initial_l23", "transfer_rate_l23", "referral_share_l45")


def starting_values(theta=None):
    """Workbook values of the Stage 1 and 3-5 parameters (Stage 2 is solved), and the
    fixed Rescue share of L2/3 transfers received at L5."""
    values = {}
    for county in CALIBRATED_COUNTIES:
        param = parameter_draw(county)
        values[county] = {
            **{(name, None): param[name] for name in (
                "PT_scale", "p_elec_CS|highrisk", "p_elec_CS|preterm", *complications.INCIDENCE,
                *transfers.FITTED.values())},
            **{("p_cs_capacity", index): param["p_cs_capacity"][index] for index in CS_CAPACITY.values()},
            ("p_l5_given_l23_transfer", None): RESCUE["to_l5"] / RESCUE["with_destination"],
        }
        if theta is not None:  # Sensitivity: another OR_recognized_L45.
            values[county][("OR_recognized_L45", None)] = theta
    return values


def update_preterm(values, fits, targets, pt_damping):
    """Stage 1: PT_scale from a round's simulations."""
    new = {county: dict(values[county]) for county in CALIBRATED_COUNTIES}
    for county in CALIBRATED_COUNTIES:
        new[county][("PT_scale", None)] = preterm.update(
            values[county][("PT_scale", None)], targets[county]["preterm_fac"],
            fits[county]["metrics"]["preterm_fac"], pt_damping)
    return new


def route_county(county, values, tau):
    """Stage 2 closed form for one county (stages/stage2_routing.py), given its calibrated values
    (PT_scale and p_ANC_base set the recognition rate r; OR_recognized_L45 is theta)
    and transfer share tau. Births by final location keep the county total."""
    param = calculate_derived_parameters(apply_values(parameter_draw(county), values))
    dhs = DHS[county]
    home, final_l23, l5_share = observed_locations(county)
    solved = routing.solve_county(
        anc=dhs["anc4"], home_noanc=dhs["home_noanc"], home_anc=dhs["home_anc"], final_l23=final_l23,
        r=routing.recognition_rate(param), theta=param["OR_recognized_L45"], tau=tau,
    )
    total = float(np.sum(param["base_LB"]))
    shares = (home, solved["final_l23"], solved["final_l45"] * (1 - l5_share), solved["final_l45"] * l5_share)
    return {
        **{(name, None): solved[name] for name in ROUTING_PARAMETERS},
        ("p_ANC_base", None): dhs["anc4"],
        **{("base_LB", index): total * share for index, share in enumerate(shares)},
    }


def solve_routing(values, tau, max_iterations=10):
    """Stage 2 repeated until stable: p_ANC_base and base_LB feed the recognition rate."""
    values = {county: dict(values[county]) for county in CALIBRATED_COUNTIES}
    for _ in range(max_iterations):
        change = 0.0
        for county in CALIBRATED_COUNTIES:
            new = route_county(county, values[county], tau[county])
            change = max(change, *(abs(value - values[county].get(key, np.inf)) for key, value in new.items()))
            values[county].update(new)
        if change < 1e-8:
            return values
    raise RuntimeError("Stage 2 routing did not converge")


def fit_transfers(county, values, targets, mix, args):
    """Stage 3 counting solution for one county: the new values, and whether a
    probability ended at 0 or 1."""
    names = list(transfers.FITTED.values())
    x = np.array([values[(name, None)] / 100 for name in names])  # the workbook stores percentages
    for _ in range(args.max_iterations):
        current = {**values, **{(name, None): 100 * value for name, value in zip(names, x)}}
        counts = transfers.pools(pd.concat(simulate(county, current, args.runs, args.months, args.base_seed), ignore_index=True))
        if args.tau_target is not None:
            # Sensitivity: tau fixed, Rescue reason mix (see transfers.solve_tau_mix).
            updated, at_bound = transfers.solve_tau_mix(x[2], counts, args.tau_target, mix)
        else:
            t_preterm, t_notsevere, at_bound = transfers.solve(counts, targets, start=np.clip(x[:2], 0.01, 0.99))
            t_predicted = transfers.updated_predicted(x[2], counts, t_preterm, t_notsevere, mix["labour_emergency"])
            updated = (t_preterm, t_notsevere, t_predicted)
        change = float(np.max(np.abs(np.array(updated) - x)))
        x = np.array(updated)
        if change < args.tolerance:
            break
    return {**values, **{(name, None): 100 * value for name, value in zip(names, x)}}, at_bound


def measure_county(county, values, targets, at_bound, args):
    """The round's final simulation for one county: model values for the round report
    and the counts that Stages 1, 4 and 5 use."""
    people = pd.concat(simulate(county, values, args.runs, args.months, args.base_seed), ignore_index=True)
    metrics = summarize(people)
    reported = (*targets, *DIAGNOSTICS)
    report = {
        "county": county,
        **{name: value for (name, index), value in values.items() if index is None},
        "at_bound": at_bound,
        **{f"{name}_model": metrics[name] for name in reported},
        **{f"{name}_target": targets.get(name, np.nan) for name in reported},
    }
    return {"values": values, "metrics": metrics, "report": report,
            "planned_cs": planned_cs.coefficients(people), "cs_l4": complications.cs_l4_components(people)}


def update_stages_4_5(values, fits, targets):
    """Planned CS (Stage 4) and complication and CS capacity values (Stage 5) from
    this round's simulations."""
    new = {county: dict(values[county]) for county in CALIBRATED_COUNTIES}
    highrisk, preterm = planned_cs.solve_shared_preterm(
        {c: fits[c]["planned_cs"] for c in CALIBRATED_COUNTIES},
        {c: targets[c]["elective_cs_l4"] for c in CALIBRATED_COUNTIES},
        targets[CALIBRATED_COUNTIES[0]]["planned_cs_preterm"],  # shared by the counties
    )
    for county in CALIBRATED_COUNTIES:
        new[county][("p_elec_CS|highrisk", None)] = highrisk[county]
        new[county][("p_elec_CS|preterm", None)] = preterm
        stage5 = {name: values[county][(name, None)] for name in complications.INCIDENCE}
        stage5.update({key: values[county][("p_cs_capacity", index)] for key, index in CS_CAPACITY.items()})
        stage5 = complications.updated(stage5, fits[county]["metrics"], targets[county], fits[county]["cs_l4"])
        new[county].update({(name, None): stage5[name] for name in complications.INCIDENCE})
        new[county].update({("p_cs_capacity", index): stage5[key] for key, index in CS_CAPACITY.items()})
    return new


def largest_changes(old, new, tau, new_tau):
    """Largest change across counties in each group of values."""
    relative = lambda keys: max(abs(new[c][k] / old[c][k] - 1) for c in CALIBRATED_COUNTIES for k in keys if old[c][k] > 0)
    absolute = lambda keys: max(abs(new[c][k] - old[c][k]) for c in CALIBRATED_COUNTIES for k in keys)
    return {
        "tau": max(abs(new_tau[c] - tau[c]) for c in CALIBRATED_COUNTIES),
        "PT_scale (relative)": relative([("PT_scale", None)]),
        "planned CS": absolute([("p_elec_CS|highrisk", None), ("p_elec_CS|preterm", None)]),
        "Stage 5 (relative)": relative(STAGE_5),
    }


# A round with every change below its tolerance ends the loop. Uterine rupture and
# sepsis are rare, so the short in-loop simulations resolve Stage 5 to about 2%.
TOLERANCES = {"tau": 0.005, "PT_scale (relative)": 0.01, "planned CS": 0.005, "Stage 5 (relative)": 0.03}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="Calibration/results/stages_1_to_5")
    # Runs = parameter draws; calibration targets the average over draws.
    parser.add_argument("--runs", type=int, default=DEFAULTS["loop_runs"])
    parser.add_argument("--months", type=int, default=DEFAULTS["loop_months"])
    parser.add_argument("--base-seed", type=int, default=DEFAULTS["simulation_seed"])
    parser.add_argument("--max-iterations", type=int, default=6, help="Stage 3 passes per round.")
    parser.add_argument("--tolerance", type=float, default=0.002, help="Stage 3 probability change.")
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--pt-damping", type=float, default=0.5, help="Exponent on the Stage 1 correction.")
    parser.add_argument("--cause-mix", choices=("county", "pooled"), default="pooled",
                        help="Complication targets: county total x pooled cause mix, or county KHIS rates.")
    parser.add_argument("--theta", type=float, help="Sensitivity: OR_recognized_L45 for the calibrated counties.")
    parser.add_argument("--tau-target", type=float,
                        help="Sensitivity: fix tau and use the full Rescue reason mix for Stage 3.")
    args = parser.parse_args()

    output = REPO_ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    mix = transfers.known_reason_mix(RESCUE)
    targets = {county: county_targets(county, args.cause_mix) for county in CALIBRATED_COUNTIES}
    values = starting_values(args.theta)
    tau = {county: STARTING_TAU for county in CALIBRATED_COUNTIES}

    fits = None
    for round_number in range(1, args.max_rounds + 1):
        if fits is not None:
            values = update_preterm(values, fits, targets, args.pt_damping)                  # Stage 1
        values = solve_routing(values, tau)                                                  # Stage 2
        fits = {}
        for c in CALIBRATED_COUNTIES:
            fitted, at_bound = fit_transfers(c, values[c], targets[c], mix, args)          # Stage 3
            fits[c] = measure_county(c, fitted, targets[c], at_bound, args)
        values = {c: fits[c]["values"] for c in CALIBRATED_COUNTIES}
        results = pd.DataFrame([fits[c]["report"] for c in CALIBRATED_COUNTIES]).set_index("county")
        results.to_csv(output / f"round_{round_number}.csv")

        new_values = update_stages_4_5(values, fits, targets)                                # Stages 4 and 5
        new_tau = {c: fits[c]["metrics"]["transfer_rate_l23"] for c in CALIBRATED_COUNTIES}
        # Converged when the next round would change little, including its Stage 1 step.
        next_values = update_preterm(new_values, fits, targets, args.pt_damping)
        changes = largest_changes(values, next_values, tau, new_tau)
        print(f"round {round_number}: " + ", ".join(f"{name} {value:.4f}" for name, value in changes.items()),
              flush=True)
        if all(changes[name] < TOLERANCES[name] for name in TOLERANCES):
            break
        values, tau = new_values, new_tau
    else:
        print("warning: did not converge within max rounds", flush=True)

    # The saved values are those the last round was simulated with.
    save_values(values, output / "values.csv")
    results.to_csv(output / "final.csv")
    pd.set_option("display.width", 250)
    print(results.round(4).T.to_string())


if __name__ == "__main__":
    main()
