"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
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
import ChartFootnote from "@/components/results/ChartFootnote";
import KPITile from "@/components/KPITile";
import { useLocale } from "@/components/i18n/LocaleProvider";
import { WHO_KENYA_DALY_THRESHOLD_USD } from "@/lib/model-metadata";
import StoryIngredients from "@/components/results/StoryIngredients";
import IndicatorStoryCharts from "@/components/results/IndicatorStoryCharts";
import { shouldShowKpi, shouldShowStory } from "@/lib/indicators";
import {
  chartLegendProps,
  chartMargins,
  chartMarginsWithLegend,
  chartTooltipProps,
  xAxisLabel,
  yAxisLabel,
} from "@/lib/chart-labels";
import { ScenarioResult } from "@/lib/scenarios";

interface Props {
  result: ScenarioResult;
  selectedIndicators: Set<string>;
}

export default function ResultsStories({ result, selectedIndicators }: Props) {
  const { t, locale } = useLocale();
  const { summary, timeseries, cost_breakdown, deaths_by_cause, resource_adequacy_end_of_run } =
    result;

  const ciLower = timeseries.maternal_mortality_rate.ci_lower;
  const ciUpper = timeseries.maternal_mortality_rate.ci_upper;
  const showCi = !!(ciLower && ciUpper && ciLower.length === timeseries.months.length);

  const mmData = timeseries.months.map((m, i) => ({
    month: m,
    baseline: timeseries.maternal_mortality_rate.baseline[i],
    intervention: timeseries.maternal_mortality_rate.intervention[i],
    ...(showCi ? { ciBand: [ciLower![i], ciUpper![i]] as [number, number] } : {}),
  }));

  const deliveryData = timeseries.months.map((m, i) => ({
    month: m,
    l4: timeseries.delivery_location.intervention.l4[i],
    l5: timeseries.delivery_location.intervention.l5[i],
    home: timeseries.delivery_location.intervention.home[i],
    l23: timeseries.delivery_location.intervention.l23[i],
  }));

  const costEffective = summary.cost_effectiveness_ratio_to_threshold < 1;
  const threshold = WHO_KENYA_DALY_THRESHOLD_USD.toLocaleString();

  const showKpi = shouldShowStory("kpi", selectedIndicators);
  const showStory01 = shouldShowStory("story01", selectedIndicators);
  const showStory02 = shouldShowStory("story02", selectedIndicators);
  const showStory03 = shouldShowStory("story03", selectedIndicators);
  const showStory04 = shouldShowStory("story04", selectedIndicators);

  const noStories =
    !showStory01 && !showStory02 && !showStory03 && !showStory04 && !showKpi;

  return (
    <div className="space-y-12">
      {showKpi && (
        <section>
          <StoryIngredients story="kpi" selected={selectedIndicators} />
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {shouldShowKpi("deaths", selectedIndicators) && (
            <KPITile
              label={t("kpi.deaths")}
              value={summary.maternal_deaths_averted.toLocaleString(undefined, {
                maximumFractionDigits: 0,
              })}
            />
          )}
          {shouldShowKpi("dalys", selectedIndicators) && (
            <KPITile
              label={t("kpi.dalys")}
              value={summary.dalys_averted.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            />
          )}
          {shouldShowKpi("costPerDaly", selectedIndicators) && (
            <KPITile
              label={t("kpi.costPerDaly")}
              value={`$${summary.cost_per_daly_averted_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
              accent
            />
          )}
          {shouldShowKpi("totalCost", selectedIndicators) && (
            <KPITile
              label={t("kpi.totalCost")}
              value={`$${summary.cumulative_cost_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
            />
          )}
          {shouldShowKpi("severe", selectedIndicators) && (
            <KPITile
              label={t("kpi.severeOutcomes")}
              value={summary.severe_maternal_outcomes_averted.toLocaleString(undefined, {
                maximumFractionDigits: 0,
              })}
            />
          )}
          </div>
        </section>
      )}

      {noStories && (
        <div className="bg-paper-deep border border-border rounded-lg p-8 text-center text-ink-muted">
          {t("stories.noIndicators")}
        </div>
      )}

      {showStory01 && (
        <section className="bg-card border border-border rounded-xl p-8">
          <div className="text-[11px] uppercase tracking-widest text-ink-muted mb-2">
            {t("stories.story01")}
          </div>
          <h2 className="font-display text-2xl mb-2 editorial-underline inline">
            {t("stories.worthIt")}
          </h2>
          <StoryIngredients story="story01" selected={selectedIndicators} />
          {locale === "en" ? (
            <p className="text-ink-soft mb-6 max-w-2xl">{result.narrative.in_plain_english}</p>
          ) : (
            <details className="mb-6 text-sm border border-border-soft rounded-lg px-4 py-3 bg-paper-deep/40">
              <summary className="cursor-pointer text-ink-muted hover:text-ink">
                {t("exec.englishDetail")}
              </summary>
              <p className="mt-3 text-ink-soft leading-relaxed">{result.narrative.in_plain_english}</p>
            </details>
          )}
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <div className="font-display text-5xl text-accent num mb-2">
                $
                {summary.cost_per_daly_averted_usd.toLocaleString(undefined, {
                  maximumFractionDigits: 0,
                })}
              </div>
              <p className="text-sm text-ink-muted">
                {t("stories.perDalyVs", { threshold })}
                {costEffective ? t("stories.costEffective") : t("stories.aboveThreshold")}
              </p>
            </div>
            <ChartPanel
              chartId="cost-breakdown"
              title={t("charts.costBreakdown")}
              filename="cost-breakdown"
              height={260}
              showTitle
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={cost_breakdown} margin={chartMargins}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2DAC8" />
                  <XAxis
                    dataKey="category"
                    tick={{ fontSize: 10 }}
                    label={xAxisLabel(t("charts.costCategory"))}
                  />
                  <YAxis tick={{ fontSize: 10 }} label={yAxisLabel(t("charts.costUsd"))} />
                  <Tooltip {...chartTooltipProps({ valueKind: "currency" })} />
                  <Bar dataKey="amount_usd" fill="#2E5F5C" />
                </BarChart>
              </ResponsiveContainer>
            </ChartPanel>
            <ChartFootnote>
              {t("stories.footnoteCost", { threshold })}
            </ChartFootnote>
          </div>
        </section>
      )}

      {showStory02 && (
        <section className="bg-card border border-border rounded-xl p-8">
          <div className="text-[11px] uppercase tracking-widest text-ink-muted mb-2">
            {t("stories.story02")}
          </div>
          <h2 className="font-display text-2xl mb-2">{t("stories.mothers")}</h2>
          <StoryIngredients story="story02" selected={selectedIndicators} />
          <ChartPanel
            chartId="maternal-mortality"
            title={t("charts.mmrTitle")}
            filename="maternal-mortality"
            height={340}
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mmData} margin={chartMarginsWithLegend}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2DAC8" />
                <XAxis dataKey="month" tick={{ fontSize: 10 }} label={xAxisLabel(t("charts.month"))} />
                <YAxis
                  tick={{ fontSize: 10 }}
                  label={yAxisLabel(t("charts.mmr"))}
                />
                <Tooltip {...chartTooltipProps({ valueKind: "mmr", labelPrefix: t("charts.month") })} />
                <Legend {...chartLegendProps} />
                {showCi && (
                  <Area
                    type="monotone"
                    dataKey="ciBand"
                    stroke="none"
                    fill="#2E5F5C"
                    fillOpacity={0.14}
                    name="95% interval (intervention)"
                    legendType="rect"
                    isAnimationActive={false}
                  />
                )}
                <Line
                  type="monotone"
                  dataKey="baseline"
                  stroke="#9C9082"
                  name="Baseline MMR"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="intervention"
                  stroke="#2E5F5C"
                  name="Intervention MMR"
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartPanel>
          <ChartFootnote>
            {showCi
              ? t("stories.footnoteMmrCi", { runs: result.meta.n_runs })
              : t("stories.footnoteMmrQuick")}
          </ChartFootnote>
          <div className="mt-6 grid md:grid-cols-2 gap-3">
            {deaths_by_cause.slice(0, 4).map((d) => (
              <div
                key={d.cause}
                className="flex justify-between text-sm border-b border-border-soft py-2"
              >
                <span>{d.cause}</span>
                <span className="num text-positive">
                  {t("stories.reduction", { pct: d.percent_reduction })}
                </span>
              </div>
            ))}
          </div>
          <IndicatorStoryCharts
            result={result}
            indicatorIds={[
              "cs_rate",
              "normal_referral",
              "emergency_transfer",
              "high_risk_pregnancy",
              "maternal_complication_rate",
              "severe_maternal_outcomes",
            ]}
            selected={selectedIndicators}
          />
        </section>
      )}

      {showStory03 && (
        <section className="bg-card border border-border rounded-xl p-8">
          <div className="text-[11px] uppercase tracking-widest text-ink-muted mb-2">
            {t("stories.story03")}
          </div>
          <h2 className="font-display text-2xl mb-2">{t("stories.delivery")}</h2>
          <StoryIngredients story="story03" selected={selectedIndicators} />
          <ChartPanel
            chartId="delivery-location"
            title={t("charts.deliveryTitle")}
            filename="delivery-location"
            height={340}
          >
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={deliveryData} margin={chartMarginsWithLegend}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2DAC8" />
                <XAxis dataKey="month" tick={{ fontSize: 10 }} label={xAxisLabel(t("charts.month"))} />
                <YAxis
                  tick={{ fontSize: 10 }}
                  unit="%"
                  label={yAxisLabel(t("charts.shareBirths"))}
                />
                <Tooltip {...chartTooltipProps({ valueKind: "percent", labelPrefix: t("charts.month") })} />
                <Legend {...chartLegendProps} />
                <Area type="monotone" dataKey="l4" stackId="1" stroke="#2E5F5C" fill="#2E5F5C" name="L4" />
                <Area type="monotone" dataKey="l5" stackId="1" stroke="#B5471F" fill="#B5471F" name="L5" />
                <Area type="monotone" dataKey="l23" stackId="1" stroke="#7E7464" fill="#7E7464" name="L2/3" />
                <Area type="monotone" dataKey="home" stackId="1" stroke="#9C9082" fill="#9C9082" name="Home" />
              </AreaChart>
            </ResponsiveContainer>
          </ChartPanel>
          <ChartFootnote>{t("stories.footnoteDelivery")}</ChartFootnote>
          <IndicatorStoryCharts
            result={result}
            indicatorIds={["anc_coverage", "anc_rate"]}
            selected={selectedIndicators}
          />
        </section>
      )}

      {showStory04 && (
        <section className="bg-card border border-border rounded-xl p-8">
          <div className="text-[11px] uppercase tracking-widest text-ink-muted mb-2">
            {t("stories.story04")}
          </div>
          <h2 className="font-display text-2xl mb-2">{t("stories.coping")}</h2>
          <StoryIngredients story="story04" selected={selectedIndicators} />
          <div className="space-y-4">
            {resource_adequacy_end_of_run.map((r) => (
              <div key={r.name}>
                <div className="flex justify-between text-sm mb-1">
                  <span>{r.name}</span>
                  <span className="num">{t("stories.adequate", { pct: r.percent })}</span>
                </div>
                <div className="h-2 bg-paper-deep rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      r.status === "positive"
                        ? "bg-positive"
                        : r.status === "warning"
                          ? "bg-warning"
                          : "bg-negative"
                    }`}
                    style={{ width: `${Math.min(r.percent, 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
          <IndicatorStoryCharts
            result={result}
            indicatorIds={["equipment_capacity", "supply_capacity"]}
            selected={selectedIndicators}
          />
        </section>
      )}
    </div>
  );
}
