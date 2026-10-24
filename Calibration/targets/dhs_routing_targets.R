# DHS 2022 delivery-location and ANC inputs for routing calibration (Phase 2).
#
# Adapted from momish_dhs_chunks.R::calculate_dhs_calibration. Changes:
#   - m14 = 98 ("don't know") is missing, not 4+ ANC visits.
#   - "Other" delivery places are missing, so location shares sum to one.
#   - Kakamega (v024 = 37) is included, and a national row is added.
#   - The national ANC gradient is reported for the shared routing odds ratio.
#
# Usage:
#   Rscript Calibration/targets/dhs_routing_targets.R /path/to/KEIR8AFL.DTA \
#     Calibration/targets/dhs_routing_targets.csv
#
# Output columns: county, quantity, value, se, n_unweighted.

suppressPackageStartupMessages({
  library(haven)
  library(dplyr)
  library(survey)
})
options(survey.lonely.psu = "adjust")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: dhs_routing_targets.R KEIR8AFL.DTA output.csv")

COUNTIES <- c(kakamega = 37, kisii = 45, makueni = 17, mombasa = 1)

ir <- read_dta(args[1], col_select = c(v001, v005, v022, v024, v208, b19_01, m14_1, m15_1))

# Most recent birth within two years; m14/m15 refer to that birth.
births <- ir %>%
  filter(v208 > 0, b19_01 < 24) %>%
  mutate(
    weight = v005 / 1e6,
    anc4 = case_when(m14_1 %in% 4:20 ~ 1, m14_1 %in% 0:3 ~ 0, TRUE ~ NA_real_),
    place = case_when(
      m15_1 %in% c(10, 11, 12) ~ "home",
      m15_1 %in% c(22, 23, 26, 32, 42, 44) ~ "l23",
      m15_1 %in% c(21, 31, 41, 43) ~ "l45",
      TRUE ~ NA_character_
    ),
    home = as.numeric(place == "home"),
    l23 = as.numeric(place == "l23"),
    l45 = as.numeric(place == "l45")
  )

design <- svydesign(ids = ~v001, strata = ~v022, weights = ~weight, data = births, nest = TRUE)

estimate <- function(formula, d) {
  e <- svymean(formula, d, na.rm = TRUE)
  data.frame(quantity = names(coef(e)), value = unname(coef(e)), se = unname(SE(e)))
}

summarize_area <- function(d, county) {
  # ANC coverage uses all recent births; location shares need a known place.
  coverage <- cbind(estimate(~anc4, d), n_unweighted = sum(!is.na(d$variables$anc4)))
  d <- subset(d, !is.na(place))
  v <- d$variables
  rows <- list(
    coverage,
    cbind(estimate(~home + l23 + l45, d), n_unweighted = nrow(v))
  )
  for (status in c(0, 1)) {
    g <- subset(d, anc4 == status)
    suffix <- if (status == 1) "_anc" else "_noanc"
    e <- estimate(~home + l23 + l45, g)
    e$quantity <- paste0(e$quantity, suffix)
    rows[[length(rows) + 1]] <- cbind(e, n_unweighted = sum(v$anc4 == status, na.rm = TRUE))
  }
  out <- bind_rows(rows)
  out$quantity[out$quantity %in% c("home", "l23", "l45")] <-
    paste0(out$quantity[out$quantity %in% c("home", "l23", "l45")], "_all")
  cbind(county = county, out)
}

results <- bind_rows(
  lapply(names(COUNTIES), function(county) {
    summarize_area(subset(design, v024 == COUNTIES[[county]]), county)
  }),
  summarize_area(design, "national")
)

# ANC gradient in L2/3 use among facility births, adjusted for county.
facility <- subset(design, place %in% c("l23", "l45") & !is.na(anc4))
fit <- svyglm(l23 ~ anc4 + factor(v024), design = facility, family = quasibinomial())
log_or <- coef(fit)[["anc4"]]
results <- bind_rows(results, data.frame(
  county = "national",
  quantity = "or_l23_given_facility_anc_vs_noanc",
  value = exp(log_or),
  se = exp(log_or) * SE(fit)[["anc4"]],
  n_unweighted = nrow(facility$variables)
))

write.csv(results, args[2], row.names = FALSE)
cat("Wrote", nrow(results), "rows to", args[2], "\n")
