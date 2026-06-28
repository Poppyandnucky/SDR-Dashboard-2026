"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";
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
import KPITile from "@/components/KPITile";
import CompareSummaryTable from "@/components/compare/CompareSummaryTable";
import BackToLastResultsLink from "@/components/BackToLastResultsLink";
import ChartPanel from "@/components/export/ChartPanel";
import ResultsExportBar from "@/components/export/ResultsExportBar";
import BudgetLens from "@/components/results/BudgetLens";
import { useLocale } from "@/components/i18n/LocaleProvider";
import { CompareResponse } from "@/lib/scenarios";
import {
  getCachedComparison,
  getLastComparisonData,
  saveLastComparison,
} from "@/lib/compare-storage";
import { chartMarginsWithLegend, chartLegendProps, chartTooltipProps, xAxisLabel, yAxisLabel } from "@/lib/chart-labels";

function CompareResultsContent() {
  const { t } = useLocale();
  const searchParams = useSearchParams();
  const comparisonId = searchParams.get("comparison_id");
  const exportScopeRef = useRef<HTMLDivElement>(null);
  const [data, setData] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [sessionOnly, setSessionOnly] = useState(false);

  useEffect(() => {
    if (!comparisonId) {
      const last = getLastComparisonData();
      if (last) {
        setData(last);
        setSessionOnly(true);
      }
      setLoading(false);
      return;
    }

    let cancelled = false;
    const cached = getCachedComparison(comparisonId);

    if (cached) {
      setData(cached);
      setSessionOnly(true);
      setLoading(false);
    }

    const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
    fetch(`${API_BASE}/api/v1/scenarios/compare/${comparisonId}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d: CompareResponse | null) => {
        if (cancelled) return;
        if (d?.result_a && d?.result_b) {
          setData(d);
          saveLastComparison(d);
          setSessionOnly(false);
        } else if (!cached) {
          setData(null);
        }
      })
      .catch(() => {
        if (!cancelled && !cached) setData(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [comparisonId]);

  if (loading) {
    return <div className="p-8 text-center text-ink-muted">{t("common.loading")}</div>;
  }

  if (!data?.result_a || !data?.result_b) {
    return (
      <div className="p-8 text-center">
        <p className="text-negative mb-4">Comparison not found or expired.</p>
        <Link href="/compare" className="text-accent underline">
          {t("common.newComparison")}
        </Link>
      </div>
    );
  }

  const { result_a: a, result_b: b, combined_narrative, scenario_a, scenario_b } = data;

  const overlayData =
    a.timeseries.months.map((m, i) => ({
      month: m,
      scenarioA: a.timeseries.maternal_mortality_rate.intervention[i],
      scenarioB: b.timeseries.maternal_mortality_rate.intervention[i],
      baseline: a.timeseries.maternal_mortality_rate.baseline[i],
    })) ?? [];

  return (
    <div className="max-w-7xl mx-auto px-4 md:px-8 py-8">
      <div className="flex flex-wrap items-start justify-between gap-4 mb-8">
        <div>
          <div className="flex items-center gap-2 mb-2 flex-wrap">
            <span className="text-[11px] tracking-[0.2em] text-accent uppercase">{t("compare.label")}</span>
            {sessionOnly && (
              <span className="text-[11px] px-2 py-0.5 bg-warning/15 text-warning rounded-md">
                {t("compare.sessionSaved")}
              </span>
            )}
          </div>
          <h1 className="font-display text-4xl font-light mb-2">{t("compare.title")}</h1>
          <p className="text-ink-muted text-sm">
            <span style={{ color: "#2563A8" }}>{scenario_a.name}</span>
            {" vs "}
            <span style={{ color: "#2B7A3E" }}>{scenario_b.name}</span>
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <ResultsExportBar mode="compare" data={data} scopeRef={exportScopeRef} />
          <BackToLastResultsLink />
          <Link
            href="/compare"
            className="px-4 py-2 bg-ink text-paper rounded-md text-sm"
          >
            {t("compare.adjust")}
          </Link>
        </div>
      </div>

      {sessionOnly && (
        <div className="mb-6 text-sm bg-warning/10 border border-warning/30 rounded-lg px-4 py-3 text-ink-soft">
          {t("compare.sessionSavedDetail")}
        </div>
      )}

      {combined_narrative && (
        <p className="text-ink-soft leading-relaxed mb-8 max-w-3xl font-display text-lg">
          {combined_narrative}
        </p>
      )}

      <CompareSummaryTable data={data} />

      <BudgetLens
        mode="compare"
        labelA={scenario_a.name}
        labelB={scenario_b.name}
        costAUsd={a.summary.cumulative_cost_usd}
        costBUsd={b.summary.cumulative_cost_usd}
      />

      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
        <KPITile
          label={t("compare.deathsA")}
          value={a.summary.maternal_deaths_averted.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          sub={scenario_a.name}
        />
        <KPITile
          label={t("compare.deathsB")}
          value={b.summary.maternal_deaths_averted.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          sub={scenario_b.name}
          accent
        />
        <KPITile
          label={t("compare.costDalyA")}
          value={`$${a.summary.cost_per_daly_averted_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
        />
        <KPITile
          label={t("compare.costDalyB")}
          value={`$${b.summary.cost_per_daly_averted_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
        />
      </div>

      <div ref={exportScopeRef}>
        <section className="bg-card border border-border rounded-xl p-8 mb-12">
          <div className="flex flex-wrap items-start justify-between gap-4 mb-2">
            <div>
              <h2 className="font-display text-xl mb-2">{t("compare.overlayTitle")}</h2>
              <p className="text-sm text-ink-muted">{t("compare.overlayDesc")}</p>
            </div>
          </div>
          <ChartPanel
            chartId="comparison-mmr-overlay"
            title={t("compare.overlayTitle")}
            filename="comparison-mmr-overlay"
            height={380}
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={overlayData} margin={chartMarginsWithLegend}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2DAC8" />
                <XAxis dataKey="month" tick={{ fontSize: 10 }} label={xAxisLabel(t("charts.month"))} />
                <YAxis
                  tick={{ fontSize: 10 }}
                  label={yAxisLabel(t("charts.mmr"))}
                />
                <Tooltip {...chartTooltipProps({ valueKind: "mmr", labelPrefix: t("charts.month") })} />
                <Legend {...chartLegendProps} />
                <Line
                  type="monotone"
                  dataKey="baseline"
                  stroke="#9C9082"
                  name="Baseline"
                  dot={false}
                  strokeDasharray="4 4"
                />
                <Line
                  type="monotone"
                  dataKey="scenarioA"
                  stroke="#2563A8"
                  name={scenario_a.name}
                  dot={false}
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="scenarioB"
                  stroke="#2B7A3E"
                  name={scenario_b.name}
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartPanel>
        </section>
      </div>

      <div className="grid md:grid-cols-2 gap-6 mb-12">
        <div className="bg-paper-deep border border-border rounded-xl p-6">
          <h3 className="font-display text-lg mb-3" style={{ color: "#2563A8" }}>
            {scenario_a.name}
          </h3>
          <ul className="text-sm space-y-1 text-ink-soft">
            {a.applied_interventions.map((i) => (
              <li key={i.name}>
                • {i.name}
                {i.intensity ? ` (${i.intensity})` : ""}
              </li>
            ))}
          </ul>
        </div>
        <div className="bg-paper-deep border border-border rounded-xl p-6">
          <h3 className="font-display text-lg mb-3" style={{ color: "#2B7A3E" }}>
            {scenario_b.name}
          </h3>
          <ul className="text-sm space-y-1 text-ink-soft">
            {b.applied_interventions.map((i) => (
              <li key={i.name}>
                • {i.name}
                {i.intensity ? ` (${i.intensity})` : ""}
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-4">
        <BackToLastResultsLink variant="text" />
        <Link href="/compare" className="text-accent underline">
          {t("compare.adjustArrow")}
        </Link>
      </div>
    </div>
  );
}

export default function CompareResultsPage() {
  return (
    <Suspense fallback={<div className="p-8">Loading…</div>}>
      <CompareResultsContent />
    </Suspense>
  );
}
