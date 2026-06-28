import { WHO_KENYA_DALY_THRESHOLD_USD } from "./model-metadata";
import { TranslateFn } from "./i18n";
import { Scenario, ScenarioResult } from "./scenarios";

export type ResultsViewMode = "policy" | "analyst";

export interface ExecutiveSummaryData {
  headline: string;
  bullets: string[];
  englishNarrative?: string;
  verdict: string;
  verdictTone: "positive" | "warning" | "neutral";
  caveat: string;
  systemNote: string;
  runLabel: string;
}

function fmt(n: number): string {
  return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

export function buildExecutiveSummary(
  scenario: Scenario,
  result: ScenarioResult,
  t: TranslateFn,
  locale: "en" | "sw"
): ExecutiveSummaryData {
  const { summary } = result;
  const horizon = scenario.run.implementation_years + scenario.run.maintenance_years;
  const costEffective = summary.cost_effectiveness_ratio_to_threshold < 1;
  const resources = result.resource_adequacy_end_of_run;
  const weakest =
    resources.length > 0
      ? resources.reduce((min, r) => (r.percent < min.percent ? r : min))
      : null;

  const hasCi =
    !!result.timeseries.maternal_mortality_rate.ci_lower &&
    !!result.timeseries.maternal_mortality_rate.ci_upper;

  let caveat: string;
  if (result.meta.n_runs > 1 && hasCi) {
    caveat = t("exec.caveatRobustCi");
  } else if (result.meta.n_runs > 1) {
    caveat = t("exec.caveatRobust");
  } else {
    caveat = t("exec.caveatQuick");
  }

  let systemNote: string;
  if (!weakest) {
    systemNote = t("exec.systemNone");
  } else if (weakest.percent >= 85) {
    systemNote = t("exec.systemOk");
  } else if (weakest.percent >= 65) {
    systemNote = t("exec.systemModerate", { name: weakest.name, percent: weakest.percent });
  } else {
    systemNote = t("exec.systemStrain", { name: weakest.name, percent: weakest.percent });
  }

  const uiOnly = result.applied_interventions.filter((i) => !i.is_wired_in_model);
  if (uiOnly.length > 0) {
    caveat += ` ${t("exec.caveatUiOnly", { count: uiOnly.length })}`;
  }

  const deaths = fmt(summary.maternal_deaths_averted);
  const dalys = fmt(summary.dalys_averted);
  const totalCost = fmt(summary.cumulative_cost_usd);
  const costPerDaly = fmt(summary.cost_per_daly_averted_usd);
  const threshold = WHO_KENYA_DALY_THRESHOLD_USD.toLocaleString();

  return {
    headline: t("exec.headline", {
      years: horizon,
      deaths,
      dalys,
      totalCost,
      costPerDaly,
    }),
    bullets: [
      t("exec.bulletDeaths", { deaths, years: horizon }),
      t("exec.bulletDalysCost", { dalys, totalCost }),
      t("exec.bulletCostDaly", { costPerDaly, threshold }),
    ],
    englishNarrative: locale === "sw" ? result.narrative.in_plain_english : undefined,
    verdict: costEffective ? t("exec.verdictYes") : t("exec.verdictNo"),
    verdictTone: costEffective ? "positive" : "warning",
    caveat,
    systemNote,
    runLabel:
      result.meta.n_runs > 1
        ? t("exec.runRobust", {
            runs: result.meta.n_runs,
            seconds: result.meta.runtime_seconds.toFixed(0),
          })
        : t("exec.runQuick", { seconds: result.meta.runtime_seconds.toFixed(0) }),
  };
}
