"use client";

import { ExecutiveSummaryData } from "@/lib/results-summary";
import { ScenarioResult } from "@/lib/scenarios";
import { useLocale } from "@/components/i18n/LocaleProvider";

interface Props {
  summary: ExecutiveSummaryData;
  result: ScenarioResult;
}

export default function ExecutiveSummary({ summary, result }: Props) {
  const { t } = useLocale();

  const toneClass =
    summary.verdictTone === "positive"
      ? "bg-intervention-soft text-intervention border-intervention/20"
      : summary.verdictTone === "warning"
        ? "bg-warning/10 text-warning border-warning/30"
        : "bg-paper-deep text-ink-soft border-border";

  return (
    <section className="mb-8 bg-card border border-border rounded-xl overflow-hidden">
      <div className="px-6 py-4 border-b border-border bg-paper-deep/50 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-[11px] uppercase tracking-[0.2em] text-accent mb-1">
            {t("results.policyBrief")}
          </div>
          <h2 className="font-display text-xl">{t("exec.title")}</h2>
        </div>
        <span className="text-[11px] px-2.5 py-1 bg-card border border-border rounded-md num text-ink-muted">
          {summary.runLabel}
        </span>
      </div>

      <div className="p-6 md:p-8">
        <p className="font-display text-lg text-ink-soft leading-relaxed mb-6 max-w-3xl">
          {summary.headline}
        </p>

        {summary.englishNarrative && (
          <details className="mb-6 text-sm border border-border-soft rounded-lg px-4 py-3 bg-paper-deep/40">
            <summary className="cursor-pointer text-ink-muted hover:text-ink">
              {t("exec.englishDetail")}
            </summary>
            <p className="mt-3 text-ink-soft leading-relaxed">{summary.englishNarrative}</p>
          </details>
        )}

        <ul className="space-y-2 mb-6">
          {summary.bullets.map((bullet) => (
            <li key={bullet} className="flex gap-2 text-sm text-ink">
              <span className="text-accent shrink-0">•</span>
              <span>{bullet}</span>
            </li>
          ))}
        </ul>

        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
          <SummaryStat
            label={t("exec.deathsAverted")}
            value={result.summary.maternal_deaths_averted.toLocaleString(undefined, {
              maximumFractionDigits: 0,
            })}
          />
          <SummaryStat
            label={t("exec.dalysAverted")}
            value={result.summary.dalys_averted.toLocaleString(undefined, {
              maximumFractionDigits: 0,
            })}
          />
          <SummaryStat
            label={t("exec.costPerDaly")}
            value={`$${result.summary.cost_per_daly_averted_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
            accent
          />
          <SummaryStat
            label={t("exec.totalCost")}
            value={`$${result.summary.cumulative_cost_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
          />
        </div>

        <div className="flex flex-wrap gap-2 mb-4">
          <span className={`text-xs px-3 py-1.5 rounded-md border font-medium ${toneClass}`}>
            {summary.verdict}
          </span>
          <span className="text-xs px-3 py-1.5 rounded-md border border-border bg-paper-deep text-ink-soft">
            {summary.systemNote}
          </span>
        </div>

        {result.applied_interventions.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-4">
            {result.applied_interventions.map((i) => (
              <span
                key={i.name}
                className={`px-2 py-0.5 rounded text-[11px] ${
                  i.is_wired_in_model
                    ? "bg-intervention-soft/60 text-intervention"
                    : "bg-accent-soft/50 text-accent border border-accent/20"
                }`}
              >
                {i.name}
                {!i.is_wired_in_model && ` · ${t("common.notInModelYet")}`}
              </span>
            ))}
          </div>
        )}

        <p className="text-xs text-ink-muted leading-relaxed border-t border-border-soft pt-4">
          <strong className="text-ink-soft">{t("common.caveat")}:</strong> {summary.caveat}
        </p>
      </div>
    </section>
  );
}

function SummaryStat({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div className="bg-paper-deep/60 border border-border-soft rounded-lg px-4 py-3">
      <div className="text-[10px] uppercase tracking-wider text-ink-muted mb-1">{label}</div>
      <div className={`font-display text-2xl num ${accent ? "text-accent" : "text-ink"}`}>
        {value}
      </div>
    </div>
  );
}
