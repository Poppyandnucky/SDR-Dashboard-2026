"use client";

import {
  buildReproducibilityRecord,
  CALIBRATION_NOTE,
  CALIBRATION_SCOPE,
  countyDisplayName,
  formatReproducibilityLines,
  MODEL_VERSION,
} from "@/lib/model-metadata";
import { Scenario, ScenarioResult } from "@/lib/scenarios";
import { useLocale } from "@/components/i18n/LocaleProvider";

interface Props {
  scenario: Scenario;
  result: ScenarioResult;
  runId?: string | null;
}

export default function MethodsLimitations({ scenario, result, runId }: Props) {
  const { t } = useLocale();
  const repro = buildReproducibilityRecord(scenario, result, runId);
  const reproLines = formatReproducibilityLines(repro);
  const uiOnly = result.applied_interventions.filter((i) => !i.is_wired_in_model);
  const horizon = scenario.run.implementation_years + scenario.run.maintenance_years;

  return (
    <details className="mb-8 bg-card border border-border rounded-xl overflow-hidden group">
      <summary className="px-6 py-4 cursor-pointer hover:bg-paper-deep/40 flex items-center justify-between transition list-none">
        <div className="flex items-center gap-3">
          <span className="text-ink-muted text-sm transition group-open:rotate-180">▾</span>
          <div>
            <div className="font-medium text-sm">{t("methods.title")}</div>
            <div className="text-xs text-ink-muted mt-0.5">{t("methods.subtitle")}</div>
          </div>
        </div>
        <span className="text-[11px] text-ink-muted hidden sm:inline">{MODEL_VERSION}</span>
      </summary>

      <div className="px-6 pb-6 pt-2 border-t border-border-soft text-sm text-ink-soft space-y-5">
        <section>
          <h3 className="text-xs uppercase tracking-wider text-ink-muted mb-2">{t("methods.model")}</h3>
          <p className="leading-relaxed">
            {t("methods.modelBody", {
              version: MODEL_VERSION,
              years: horizon,
              months: result.meta.n_months,
              county: countyDisplayName(scenario.county),
            })}
          </p>
        </section>

        <section>
          <h3 className="text-xs uppercase tracking-wider text-ink-muted mb-2">
            {t("methods.calibration")}
          </h3>
          <p className="leading-relaxed">
            {t("methods.calibrationBody", { scope: CALIBRATION_SCOPE, note: CALIBRATION_NOTE })}
          </p>
        </section>

        <section>
          <h3 className="text-xs uppercase tracking-wider text-ink-muted mb-2">
            {t("methods.runSettings")}
          </h3>
          <ul className="space-y-1 num text-xs">
            <li>
              {scenario.run.mode === "quick"
                ? t("methods.modeQuick", { mode: scenario.run.mode })
                : t("methods.modeRobust", { mode: scenario.run.mode, runs: result.meta.n_runs })}
            </li>
            <li>
              {t("methods.implMaint", {
                impl: scenario.run.implementation_years,
                maint: scenario.run.maintenance_years,
              })}
            </li>
            <li>{t("methods.runtime", { seconds: result.meta.runtime_seconds.toFixed(1) })}</li>
          </ul>
        </section>

        {uiOnly.length > 0 && (
          <section>
            <h3 className="text-xs uppercase tracking-wider text-warning mb-2">
              {t("methods.partialTitle")}
            </h3>
            <ul className="space-y-1 text-xs">
              {uiOnly.map((i) => (
                <li key={i.name}>• {t("methods.partialItem", { name: i.name })}</li>
              ))}
            </ul>
          </section>
        )}

        {result.meta.warnings.length > 0 && (
          <section>
            <h3 className="text-xs uppercase tracking-wider text-warning mb-2">{t("methods.warnings")}</h3>
            <ul className="space-y-1 text-xs">
              {result.meta.warnings.map((w) => (
                <li key={w}>• {w}</li>
              ))}
            </ul>
          </section>
        )}

        <section>
          <h3 className="text-xs uppercase tracking-wider text-ink-muted mb-2">
            {t("methods.limitations")}
          </h3>
          <ul className="space-y-1.5 text-xs leading-relaxed list-disc pl-4">
            <li>{t("methods.lim1")}</li>
            <li>{t("methods.lim2")}</li>
            <li>{t("methods.lim3")}</li>
            <li>{t("methods.lim4")}</li>
          </ul>
        </section>

        <section>
          <h3 className="text-xs uppercase tracking-wider text-ink-muted mb-2">{t("methods.repro")}</h3>
          <pre className="text-[11px] bg-paper-deep border border-border rounded-md p-3 overflow-x-auto num leading-relaxed">
            {reproLines.join("\n")}
          </pre>
        </section>
      </div>
    </details>
  );
}
