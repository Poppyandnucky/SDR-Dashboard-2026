"use client";

import BaselineInterventionChart from "@/components/results/BaselineInterventionChart";
import { chartsForSelection, getIndicatorSeries } from "@/lib/indicator-series";
import { ScenarioResult } from "@/lib/scenarios";

interface Props {
  result: ScenarioResult;
  indicatorIds: string[];
  selected: Set<string>;
}

export default function IndicatorStoryCharts({ result, indicatorIds, selected }: Props) {
  const specs = chartsForSelection(indicatorIds, selected);
  if (specs.length === 0) return null;

  const { months } = result.timeseries;

  return (
    <>
      {specs.map((spec) => {
        const series = getIndicatorSeries(result, spec.seriesKey);
        if (!series) return null;
        return (
          <BaselineInterventionChart
            key={spec.seriesKey}
            months={months}
            series={series}
            spec={spec}
          />
        );
      })}
    </>
  );
}
