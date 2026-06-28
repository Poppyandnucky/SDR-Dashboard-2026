import { ChartValueKind } from "./chart-labels";
import { ScenarioResult } from "./scenarios";

export type IndicatorSeriesKey =
  | "anc_rate_per_100_lb"
  | "cs_rate_per_100_lb"
  | "normal_referral_per_100_lb"
  | "emergency_transfer_per_100_lb"
  | "high_risk_per_100_lb"
  | "maternal_complication_rate_per_100_lb"
  | "severe_maternal_outcomes_per_100_lb"
  | "doppler_equipment_ratio"
  | "ctg_equipment_ratio"
  | "nurse_staff_ratio"
  | "surgical_staff_ratio";

export interface BaselineInterventionSeries {
  baseline: number[];
  intervention: number[];
}

export interface IndicatorTimeseriesBundle {
  anc_rate_per_100_lb: BaselineInterventionSeries;
  cs_rate_per_100_lb: BaselineInterventionSeries;
  normal_referral_per_100_lb: BaselineInterventionSeries;
  emergency_transfer_per_100_lb: BaselineInterventionSeries;
  high_risk_per_100_lb: BaselineInterventionSeries;
  maternal_complication_rate_per_100_lb: BaselineInterventionSeries;
  severe_maternal_outcomes_per_100_lb: BaselineInterventionSeries;
  doppler_equipment_ratio: BaselineInterventionSeries;
  ctg_equipment_ratio: BaselineInterventionSeries;
  nurse_staff_ratio: BaselineInterventionSeries;
  surgical_staff_ratio: BaselineInterventionSeries;
}

export interface IndicatorChartSpec {
  indicatorId: string;
  seriesKey: IndicatorSeriesKey;
  titleKey: string;
  yLabelKey: string;
  filename: string;
  valueKind: ChartValueKind;
  /** Multiply displayed values (e.g. ratio → percent) */
  displayScale?: number;
}

/** One chart panel per selected indicator (deduped by seriesKey within a story). */
export const INDICATOR_CHART_SPECS: Record<string, IndicatorChartSpec[]> = {
  anc_coverage: [
    {
      indicatorId: "anc_coverage",
      seriesKey: "anc_rate_per_100_lb",
      titleKey: "indicatorCharts.ancRateTitle",
      yLabelKey: "indicatorCharts.per100Lb",
      filename: "anc-rate",
      valueKind: "percent",
    },
  ],
  anc_rate: [
    {
      indicatorId: "anc_rate",
      seriesKey: "anc_rate_per_100_lb",
      titleKey: "indicatorCharts.ancRateTitle",
      yLabelKey: "indicatorCharts.per100Lb",
      filename: "anc-rate",
      valueKind: "percent",
    },
  ],
  cs_rate: [
    {
      indicatorId: "cs_rate",
      seriesKey: "cs_rate_per_100_lb",
      titleKey: "indicatorCharts.csRateTitle",
      yLabelKey: "indicatorCharts.per100Lb",
      filename: "cs-rate",
      valueKind: "percent",
    },
  ],
  normal_referral: [
    {
      indicatorId: "normal_referral",
      seriesKey: "normal_referral_per_100_lb",
      titleKey: "indicatorCharts.referralTitle",
      yLabelKey: "indicatorCharts.per100Lb",
      filename: "normal-referral",
      valueKind: "default",
    },
  ],
  emergency_transfer: [
    {
      indicatorId: "emergency_transfer",
      seriesKey: "emergency_transfer_per_100_lb",
      titleKey: "indicatorCharts.emergencyTransferTitle",
      yLabelKey: "indicatorCharts.per100Lb",
      filename: "emergency-transfer",
      valueKind: "default",
    },
  ],
  high_risk_pregnancy: [
    {
      indicatorId: "high_risk_pregnancy",
      seriesKey: "high_risk_per_100_lb",
      titleKey: "indicatorCharts.highRiskTitle",
      yLabelKey: "indicatorCharts.per100Lb",
      filename: "high-risk",
      valueKind: "percent",
    },
  ],
  maternal_complication_rate: [
    {
      indicatorId: "maternal_complication_rate",
      seriesKey: "maternal_complication_rate_per_100_lb",
      titleKey: "indicatorCharts.complicationRateTitle",
      yLabelKey: "indicatorCharts.per100Lb",
      filename: "complication-rate",
      valueKind: "default",
    },
  ],
  severe_maternal_outcomes: [
    {
      indicatorId: "severe_maternal_outcomes",
      seriesKey: "severe_maternal_outcomes_per_100_lb",
      titleKey: "indicatorCharts.severeOutcomesTitle",
      yLabelKey: "indicatorCharts.per100Lb",
      filename: "severe-outcomes",
      valueKind: "default",
    },
  ],
  equipment_capacity: [
    {
      indicatorId: "equipment_capacity",
      seriesKey: "doppler_equipment_ratio",
      titleKey: "indicatorCharts.dopplerRatioTitle",
      yLabelKey: "indicatorCharts.ratioActualNeeded",
      filename: "doppler-ratio",
      valueKind: "default",
      displayScale: 100,
    },
    {
      indicatorId: "equipment_capacity",
      seriesKey: "ctg_equipment_ratio",
      titleKey: "indicatorCharts.ctgRatioTitle",
      yLabelKey: "indicatorCharts.ratioActualNeeded",
      filename: "ctg-ratio",
      valueKind: "default",
      displayScale: 100,
    },
  ],
  supply_capacity: [
    {
      indicatorId: "supply_capacity",
      seriesKey: "nurse_staff_ratio",
      titleKey: "indicatorCharts.nurseRatioTitle",
      yLabelKey: "indicatorCharts.ratioActualNeeded",
      filename: "nurse-ratio",
      valueKind: "default",
      displayScale: 100,
    },
    {
      indicatorId: "supply_capacity",
      seriesKey: "surgical_staff_ratio",
      titleKey: "indicatorCharts.surgicalRatioTitle",
      yLabelKey: "indicatorCharts.ratioActualNeeded",
      filename: "surgical-ratio",
      valueKind: "default",
      displayScale: 100,
    },
  ],
};

export function getIndicatorSeries(
  result: ScenarioResult,
  key: IndicatorSeriesKey
): BaselineInterventionSeries | null {
  const bundle = result.timeseries.indicator_series;
  if (!bundle) return null;
  return bundle[key] ?? null;
}

/** Charts to render for selected indicators in a story, deduped by seriesKey. */
export function chartsForSelection(
  indicatorIds: string[],
  selected: Set<string>
): IndicatorChartSpec[] {
  const seen = new Set<IndicatorSeriesKey>();
  const out: IndicatorChartSpec[] = [];

  for (const id of indicatorIds) {
    if (!selected.has(id)) continue;
    const specs = INDICATOR_CHART_SPECS[id];
    if (!specs) continue;
    for (const spec of specs) {
      if (seen.has(spec.seriesKey)) continue;
      seen.add(spec.seriesKey);
      out.push(spec);
    }
  }

  return out;
}
