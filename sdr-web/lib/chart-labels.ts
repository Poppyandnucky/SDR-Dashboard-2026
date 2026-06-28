/** Shared Recharts axis label styling */
export const axisLabelStyle = { fontSize: 11, fill: "#7E7464" };

/** Charts without a legend */
export const chartMargins = { top: 12, right: 16, left: 12, bottom: 40 };

/** Charts with a top legend — extra top for legend, extra bottom for x-axis label */
export const chartMarginsWithLegend = { top: 40, right: 16, left: 12, bottom: 44 };

export type ChartValueKind = "default" | "percent" | "currency" | "mmr" | "count";

/** Round hover values for readable, decision-ready tooltips */
export function formatChartValue(value: unknown, kind: ChartValueKind = "default"): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";

  switch (kind) {
    case "percent":
      return `${n.toFixed(1)}%`;
    case "currency":
      if (Math.abs(n) >= 1_000_000) {
        return `$${(n / 1_000_000).toFixed(2)}M`;
      }
      if (Math.abs(n) >= 10_000) {
        return `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
      }
      return `$${n.toLocaleString(undefined, { maximumFractionDigits: 0, minimumFractionDigits: 0 })}`;
    case "mmr":
      return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
    case "count":
      return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
    default:
      return formatSmartNumber(n);
  }
}

function formatSmartNumber(n: number): string {
  const abs = Math.abs(n);
  if (abs >= 1_000) {
    return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
  }
  if (abs >= 100) {
    return n.toLocaleString(undefined, { maximumFractionDigits: 1 });
  }
  if (abs >= 1) {
    return n.toLocaleString(undefined, { maximumFractionDigits: 1 });
  }
  if (abs >= 0.01) {
    return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  return n.toPrecision(2);
}

export const tooltipContentStyle = {
  fontSize: 12,
  borderRadius: 8,
  border: "1px solid #E2DAC8",
  backgroundColor: "#FFFFFF",
  boxShadow: "0 4px 12px rgba(28, 26, 21, 0.08)",
};

export const tooltipLabelStyle = { color: "#4A4339", fontWeight: 600, marginBottom: 4 };

export const tooltipItemStyle = { color: "#1C1A15" };

/** Props to spread onto Recharts `<Tooltip />` */
export function chartTooltipProps(options: {
  valueKind?: ChartValueKind;
  /** e.g. "Month" → "Month 12" */
  labelPrefix?: string;
  /** For bar charts with category labels */
  labelSuffix?: string;
}) {
  const { valueKind = "default", labelPrefix, labelSuffix } = options;

  return {
    formatter: (value: number, name: string) => [
      formatChartValue(value, valueKind),
      name,
    ] as [string, string],
    labelFormatter: (label: string | number) => {
      const text = String(label);
      if (labelPrefix) return `${labelPrefix} ${text}`;
      if (labelSuffix) return `${text}${labelSuffix}`;
      return text;
    },
    contentStyle: tooltipContentStyle,
    labelStyle: tooltipLabelStyle,
    itemStyle: tooltipItemStyle,
  };
}

export function xAxisLabel(text: string) {
  return {
    value: text,
    position: "bottom" as const,
    offset: 8,
    style: axisLabelStyle,
  };
}

export function yAxisLabel(text: string) {
  return {
    value: text,
    angle: -90,
    position: "insideLeft" as const,
    offset: 10,
    style: { ...axisLabelStyle, textAnchor: "middle" as const },
  };
}

/** Legend above plot area so it does not overlap the x-axis label */
export const chartLegendProps = {
  verticalAlign: "top" as const,
  align: "right" as const,
  iconType: "line" as const,
  wrapperStyle: { fontSize: 11, lineHeight: "18px", top: 0, right: 0 },
};
