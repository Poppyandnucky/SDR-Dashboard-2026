# MOMISH calibration workflow

This calibration covers the three MOMISH counties: **Kisii, Makueni, and
Mombasa** (`CALIBRATED_COUNTIES` in `calibration_common.py`). Kakamega keeps its
existing values and is neither calibrated nor validated here.

Principles:

- Every target comes from a script in `Calibration/targets/` with a stated
  source. Raw data stay outside the repository; scripts take their paths as
  arguments. The numbers the calibration uses are written out (as counts where
  possible) in `calibration_data.py`; the scripts' CSV outputs are not
  committed.
- Each stage is solved by explicit equations. Stages whose targets are measured
  after emergency transfer are solved together by repeating those equations
  until nothing changes (a fixed-point loop). No black-box optimizer is used.
- Code never modifies `SDR Parameters.xlsx`. Calibrated values are applied to
  each parameter draw in memory, and the calibration ends with one **values
  table** (`values.csv`: sheet, parameter, county, index, value, and the current
  workbook value) for manual entry.
- Outcomes that are not fitted are reported as validation checks.

`Calibration/phase1_anc_routing_supplementary_methods.tex` is the full methods
text.

## Running

Build the targets (raw paths are on the MOMISH OneDrive):

```bash
Rscript Calibration/targets/dhs_routing_targets.R /path/to/KEIR8AFL.DTA Calibration/targets/dhs_routing_targets.csv
python Calibration/targets/khis_level_targets.py /path/to/L23_ProportionLB_20260430.xlsx Calibration/targets/khis_level_targets.csv
python Calibration/targets/momish_khis_targets.py "/path/to/Kisii and Mombasa Data/3rd version" Calibration/targets/momish_khis_targets.csv
python Calibration/targets/rescue_transfer_rate.py "/path/to/Facility Readiness - Rescue_referrals.xlsx" \
    /path/to/L23_ProportionLB_20260430.xlsx Calibration/targets/rescue_transfer_rate.csv
```

After rebuilding a CSV, copy the changed numbers into `calibration_data.py`.

Recalibrate Stages 1-6 and validate (writes a timestamped folder under
`Calibration/results/` with `values.csv` and `target_validation.csv`):

```bash
python Calibration/run_calibration.py recalibrate
```

Validate the current production workbook without recalibrating:

```bash
python Calibration/run_calibration.py reproduce
```

Stages 1-5 can also be run directly with
`python Calibration/calibrate_stages_1_to_5.py --output-dir <folder>`, which
writes `values.csv`; Stage 6 adds its two values to `values.csv` with
`python Calibration/calibrate_stage6_mortality.py --stage-dir <folder>`.
Validation of the workbook with those values applied:
`python Calibration/validate_baseline_calibration.py --values <folder>/values.csv`.
Scripts read `SDR Parameters.xlsx` in the repository; set `SDR_PARAMS_PATH` to
use another workbook.

## Stages

Notation: "initial" location is where labour starts; "final" location is where
the woman delivers after any emergency transfer (what DHS and KHIS record).
tau is the share of women starting labour at L2/3 who are transferred.

| Stage | Fitted (per county unless noted) | Targets | Method |
| ---: | --- | --- | --- |
| 1 Preterm | `PT_scale` | `preterm_fac_target` = 0.134 among all facility births after transfer (Waiswa et al. 2020) | `PT_scale <- PT_scale x (target / modeled)^0.5` in the loop |
| 2 Routing | `home_noANC`, `home_lowrisk`, `baseline_l23_preference`; derived `p_lb_l23_45`, `base_LB` | DHS 2022 home share by ANC status; KHIS 2022 L2/3 share of facility births (final) | Closed form given the recognition rate and tau; `OR_recognized_L45` = 2 shared |
| 3 Transfers | `t_l23_l45_preterm`, `t_l23_l45_notsevere`, `t_l23_l45_predicted`, `p_l5_given_l23_transfer` | KHIS `p_preterm_l23`; `p_mcr_l23_l4` (L2/3 vs internal L4); Rescue labour-emergency share 58.3%; Rescue receiving level 58% L5 | Counting equations re-simulated until stable |
| 4 Planned CS | `p_elec_CS|highrisk` (county); `p_elec_CS|preterm` (shared) | `elective_cs_l4_target` = 0.40 x `cs_l4_target` at final L4 (Sandie et al. 2023); `planned_cs_preterm_target` = 0.265 x 1232/4477 = 0.0729 (Ghosh et al. 2022; Adeniran et al. 2024) | Closed form, 4 equations in 4 unknowns |
| 5 Complications and CS capacity | `p_aph`, `p_ruptured_uterus`, `p_eclampsia`, `p_OL_scale`, `pph_incidence_multiplier`, `sepsis_incidence_multiplier`, `p_cs_capacity` [L2/3, L4] | Six internal-L4 complication targets = county `p_mcr_l4` (KHIS x 4.48) x pooled three-county cause share; `cs_l4_target`, `p_cs_l23` (raw KHIS) | `x <- x x target / modeled` in the loop |
| 6 Mortality | `p_mat_death_facility_baseline`, `RR_MM_l23_vs_l4` (one value shared by the three counties) | KHIS deaths pooled over the three counties, times 0.618 (direct, non-abortion deaths; Say et al. 2014, as for Kakamega): L4 84 / 168,887 -> 30.8 and L2/3 7 / 60,657 -> 7.1 per 100,000 (January 2022-June 2025); each county checked against its exact Poisson 95% interval | `A = T_L4 / M_L4`; `p_mat_death_facility_baseline <- x A`; `RR_MM_l23_vs_l4 <- x (T_L23 / M_L23) / A`, run once after the loop |

Every simulation in the loop, in Stage 6, and in validation gives each run its
own draw of the sampled parameters (seed 42 + run), as scenario runs do, so
targets are matched on average over parameter uncertainty. The calibrated
parameters themselves are not sampled.

Stages 1-5 run in one loop (`calibrate_stages_1_to_5.py`) because each stage's
targets depend on the others: preterm prevalence changes routing and transfers,
transfers change who delivers at L4, and planned CS and CS capacity change who
is monitored and transferred. The stages are numbered in the order each round
solves them: Stage 1 updates `PT_scale` from the previous round's simulation,
and Stages 4 and 5 are updated from the round's final Stage 3 simulation.

Held fixed: `OR_recognized_L45` = 2, `t_l23_l45_severe` = 90,
`p_comp_severe_lowrisk` = 0.08295, `p_cs_capacity` [L5], `sen_comp_trad`,
`spec_comp_trad`, and the mortality cause weights (`MM_weight_*`,
`p_MM_others`, `weight_facility_mat`).

### Target rules

- **Location.** DHS 2022 for home delivery, ANC coverage (`p_ANC_base`), and the
  national ANC gradient; KHIS 2022 for the facility-level split. Makueni County
  Referral Hospital is treated as Level 5.
- **Diagnoses.** Every recorded diagnosis (six complications, MCR, preterm) uses
  the KHIS rate x `scale_comp` (4.48). Each county keeps its own total
  complication rate (`p_mcr_l4`), but the split by cause is the pooled
  three-county mix: deaths per recorded complication are similar across
  counties (0.97-1.62 per 100), so the totals look real, whereas the cause
  split does not (Kisii records 67% of complications as obstructed labour,
  the others 25-38%). County cause-specific KHIS rates are validation checks. MOMISH KHIS January 2022-June 2025 is used for every county,
  facility-months missing from the indicator extract count as zero, and
  Mombasa's 2019-2021 numerator-only rows are excluded.
- **Procedures.** CS rates use raw KHIS.
- **L5** receives external referrals, so it is excluded from every rate target.
  External-referral mortality is handled by the reporting-only sensitivity below.

### Model changes the calibration depends on

| Change | Code |
| --- | --- |
| L4/5 capacity limit off by default (`flag_facility_capacity_limit`); it had been rationing transfers | `global_func.py`, `LB_effect.py`, `intrapartum.py` |
| L2/3 transfer destination from `p_l5_given_l23_transfer` (previously 50/50) | `intrapartum.py` |
| Preterm by ANC no longer rounded to two decimals (`odds_prob(..., decimals=None)`) | `global_func.py` |
| Live-birth shift reads the workbook baseline every month (removes a double count of transfers) | `LB_effect.py` |
| Initial routing as a baseline L2/3 preference and a recognized-risk odds ratio (`baseline_l23_preference`, `OR_recognized_L45`) | `LB_effect.py` |
| L2/3 labour-emergency transfer probability `t_l23_l45_predicted` | `intrapartum.py` |
| Pre-transfer group (`i_transfer_group`) recorded per woman | `intrapartum.py` |
| Shared facility mortality baseline `p_mat_death_facility_baseline` (staffing scales it relatively) and L2/3 relative risk `RR_MM_l23_vs_l4`; expected death probability `p_mat_death` recorded | `global_func.py`, `mortality.py` |

## Why these changes

Findings from reviewing the earlier calibration:

- The Makueni ANC-location target came from a different source from its overall
  location target (DHS vs KHIS); together they implied a negative non-ANC L2/3
  share.
- 0.122 (Waiswa et al.) is the share of L4/5 births arriving as referrals, not a
  transfer rate; an earlier target divided it by an L2/3 share.
- L4/5 `Capacity` equalled the observed L4/5 load in every county, so it rationed
  transfers.
- `p_elec_CS|preterm` had been fitted treating any CS among preterm births
  (0.265) as planned CS.
- Rounding preterm probabilities to two decimals made `PT_scale` move in steps
  of about 10%.
- PPH and sepsis targets came from literature (sepsis 15%), implying a 73-fold
  under-recording factor for sepsis; `sepsis_l4_target` also had two rows.
- KHIS facility-to-facility referral fields are empty, so transfers are inferred.

## Open issues

- **SMO** is close to reference on average over parameter draws (0.031 / 0.045 /
  0.047 vs 0.036 / 0.040 / 0.044), but single draws with a low
  `RR_comp_severe_highrisk_vs_lowrisk` give much lower SMO. Deaths are too rare
  to explain SMO; it is a check on the severity pathway.
- **Mortality anchors** are shared by the three counties, so county MMR is a
  check, not a fit. All six county-level values fall within their KHIS
  intervals; Mombasa L2/3 (8.6) is within a one-death interval (0.1-13.5).
- **L2/3 over-flagging.** At L2/3, detection is 0.8 x `sen_comp_trad` and
  0.8 x `spec_comp_trad`, so about 40% of women without a complication are
  flagged; `t_l23_l45_predicted` partly absorbs this.
- **Unanchored values.** `t_l23_l45_severe` = 90 was a search ceiling in the
  original calibration.
- **Makueni DHS ANC split** conflicts with KHIS's facility split; reported as a
  validation gap.
- **Kakamega.** The model changes (capacity limit off, unrounded preterm by
  ANC, Rescue transfer destination) apply to every county, so Kakamega's
  dashboard results shift once they are merged, although its values are
  unchanged. County-specific switches are needed if its published results
  must not move.
- **Makueni L5.** The modeled internal L5 MMR (177 per 100,000) exceeds the
  observed 75 (41-126), so in-referral cannot match it. One undocumented
  hypothesis is out-referral of complicated cases (for example to Machakos
  Level 5 or Kenyatta National Hospital); the sensitivity treats Makueni that
  way (the riskiest 1-2% of modeled L5 deliveries, or 10-15% at average proxy
  risk, match). Makueni's modeled L4 MMR is below KHIS (34 vs 45), so there is
  no sign that out-referral lowers the L4 records used in Stage 6.
- **Sepsis.** `sepsis_incidence_multiplier` is 0.03-0.04, so sepsis may be
  under-recorded by more than the common factor of 4.48.

## Sensitivity analyses

Each repeats Stages 1-6 and validation (`calibrate_stages_1_to_5.py --tau-target`,
`--theta`, `--cause-mix county`; a workbook copy with other `sen_comp_trad`/`spec_comp_trad`
via `SDR_PARAMS_PATH`).
Intervention effects are compared with `run_tau_intervention_sensitivity.py`
(40 paired runs; relative change in expected MMR).

- **L2/3 monitoring** without the 0.8 factor: tau unchanged; all targets met.
- **tau fixed at 0.15 or 0.25** with the full Rescue reason mix: modeled L2/3
  preterm doubles (0.12 vs KHIS 0.06) and the referral share of L4/5 births
  falls to 0.04-0.09, so the current rule is kept. Transfer & EMT effects are
  smaller in Makueni and Mombasa (Mombasa -4.6% vs -2.9%); combined package
  within 4 points.
- **`OR_recognized_L45` = 2.5**: parameters and county MMR change by under 2%;
  intervention effects by at most 2 points. theta = 2 is kept.
- **County cause-specific targets** instead of the pooled cause mix: incidence
  multipliers spread 4- to 9-fold; county L4 MMR fits KHIS slightly better
  (mean absolute error 16% vs 22%, weak evidence). Complication-specific effects
  shift in opposite directions in Kisii and Makueni (PPH bundle Kisii -10.3%
  pooled vs -7.4% county, Makueni -10.0% vs -13.3%); Mombasa is unchanged.

## External L5 referral sensitivity

External referrals are not included in county calibration or intervention
simulations because their volume, origin, severity, and intervention exposure
are not observed. The sensitivity is illustrative and reporting only: for a
grid of assumed external shares of L5 deliveries and late-referral odds ratios
(in-referral; for Makueni, shares of modeled L5 deliveries referred out of the
county), it computes the L5 MMR the facility would report, without adding women or
changing any modeled pathway. Two unknowns cannot be identified from one
observed MMR, so no combination is selected; the observed L5 MMR (KHIS x 0.618,
with its exact 95% interval) marks the part of the grid consistent with the
data. The mortality profile of complication-bearing internal emergency
transfers to L5 is the proxy for external referrals, with the odds ratio for
late arrival and acuity:

```
reported L5 MMR = (1 - external share) x internal L5 MMR + external share x proxy external MMR
```

```bash
python Calibration/run_external_l5_sensitivity.py --values Calibration/results/<run>/values.csv
```

## Workbook placement

Shared parameters belong in `constants`. County-specific values belong in
`county_calibrated`. Blank county cells in `county_calibrated` are ignored by
the loader and must not represent shared values. Enter each row of
`values.csv` into its sheet (`index` is the position in a list-valued
parameter, e.g. `p_cs_capacity` [L2/3, L4, L5]); rows where `value` equals
`workbook_value` need no change. After entry, `reproduce` should match the
recalibration's `target_validation.csv`.

## Files

Model: `LB_effect.py`, `global_func.py`, `intrapartum.py`, `mortality.py`,
`SDR Parameters.xlsx`.
Calibration: `Calibration/` (scripts, `stages/`, `targets/` with their CSVs,
this README, the methods `.tex`). Tests: `pytest.ini`, `tests/`.
Not migrated: `Calibration/results/`, `__pycache__`.

After migrating, enter or copy the workbook values, then run
`python -m pytest -q` and `python Calibration/run_calibration.py reproduce`, and
compare `target_validation.csv` with the recalibration run.
