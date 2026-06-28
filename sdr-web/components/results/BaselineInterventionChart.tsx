"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import ChartPanel from "@/components/export/ChartPanel";
import { useLocale } from "@/components/i18n/LocaleProvider";
import {
  BaselineInterventionSeries,
  IndicatorChartSpec,
} from "@/lib/indicator-series";
import {
  chartLegendProps,
  chartMarginsWithLegend,
  chartTooltipProps,
  ChartValueKind,
  xAxisLabel,
  yAxisLabel,
} from "@/lib/chart-labels";

interface Props {
  months: number[];
  series: BaselineInterventionSeries;
  spec: IndicatorChartSpec;
  className?: string;
}

function scaleSeries(values: number[], scale = 1): number[] {
  if (scale === 1) return values;
  return values.map((v) => v * scale);
}

export default function BaselineInterventionChart({
  months,
  series,
  spec,
  className = "mt-6",
}: Props) {
  const { t } = useLocale();
  const scale = spec.displayScale ?? 1;
  const valueKind: ChartValueKind =
    scale !== 1 && spec.valueKind === "default" ? "percent" : spec.valueKind;

  const data = months.map((month, i) => ({
    month,
    baseline: scaleSeries(series.baseline, scale)[i],
    intervention: scaleSeries(series.intervention, scale)[i],
  }));

  return (
    <div className={className}>
      <ChartPanel
        chartId={spec.filename}
        title={t(spec.titleKey)}
        filename={spec.filename}
        height={280}
        showTitle
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={chartMarginsWithLegend}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E2DAC8" />
            <XAxis
              dataKey="month"
              tick={{ fontSize: 10 }}
              label={xAxisLabel(t("charts.month"))}
            />
            <YAxis
              tick={{ fontSize: 10 }}
              label={yAxisLabel(t(spec.yLabelKey))}
            />
            <Tooltip
              {...chartTooltipProps({
                valueKind,
                labelPrefix: t("charts.month"),
              })}
            />
            <Legend {...chartLegendProps} />
            <Line
              type="monotone"
              dataKey="baseline"
              stroke="#9C9082"
              name={t("indicatorCharts.baseline")}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="intervention"
              stroke="#2E5F5C"
              name={t("indicatorCharts.intervention")}
              dot={false}
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartPanel>
    </div>
  );
}
