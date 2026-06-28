"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import ResultsExportBar from "@/components/export/ResultsExportBar";
import BackToLastComparisonLink from "@/components/BackToLastComparisonLink";
import CountyScopeBanner from "@/components/results/CountyScopeBanner";
import ExecutiveSummary from "@/components/results/ExecutiveSummary";
import MethodsLimitations from "@/components/results/MethodsLimitations";
import BudgetLens from "@/components/results/BudgetLens";
import ViewModeToggle from "@/components/results/ViewModeToggle";
import IndicatorsDrawer from "@/components/IndicatorsDrawer";
import ResultsStories from "@/components/stories/ResultsStories";
import { pollRun } from "@/lib/api";
import { getDefaultSelectedIndicators, getEssentialIndicators } from "@/lib/indicators";
import { getCachedRunResult, saveLastRun } from "@/lib/last-run-storage";
import { useLocale } from "@/components/i18n/LocaleProvider";
import { buildExecutiveSummary, ResultsViewMode } from "@/lib/results-summary";
import { DEFAULT_SCENARIO, Scenario, ScenarioResult } from "@/lib/scenarios";
import { scenarioFromURLParams, scenarioToSearchParams } from "@/lib/url-state";

function ResultsContent() {
  const { t, locale } = useLocale();
  const searchParams = useSearchParams();
  const runId = searchParams.get("run_id");
  const exportScopeRef = useRef<HTMLDivElement>(null);
  const [scenario, setScenario] = useState<Scenario | null>(null);
  const [result, setResult] = useState<ScenarioResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sessionOnly, setSessionOnly] = useState(false);
  const [viewMode, setViewMode] = useState<ResultsViewMode>("policy");
  const [selectedIndicators, setSelectedIndicators] = useState<Set<string>>(new Set());

  useEffect(() => {
    const urlScenario = scenarioFromURLParams(searchParams.get("s"));
    if (urlScenario) setScenario(urlScenario);

    if (!runId) {
      setError("No run ID found. Please run a simulation from the Design page.");
      setLoading(false);
      return;
    }

    let cancelled = false;
    const cached = getCachedRunResult(runId);

    if (cached) {
      setResult(cached.result);
      setScenario(cached.scenario);
      setSelectedIndicators(getDefaultSelectedIndicators(cached.scenario));
      setSessionOnly(true);
      setLoading(false);
    }

    pollRun(runId)
      .then((response) => {
        if (cancelled) return;
        if (response.result) {
          setResult(response.result);
          const sc = response.scenario;
          setScenario(sc);
          setSelectedIndicators(getDefaultSelectedIndicators(sc));
          saveLastRun(runId, sc, response.result);
          setSessionOnly(false);
          setError(null);
        } else if (!cached) {
          setError(
            response.error_message || "Result not available. It may have expired — please re-run."
          );
        }
      })
      .catch((e) => {
        if (cancelled) return;
        if (!cached) {
          setError(e instanceof Error ? e.message : "Failed to load results");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [runId, searchParams]);

  const executiveSummary = useMemo(
    () => (result && scenario ? buildExecutiveSummary(scenario, result, t, locale) : null),
    [scenario, result, t, locale]
  );

  const displayIndicators = useMemo(() => {
    if (!scenario) return selectedIndicators;
    if (viewMode === "policy") return getEssentialIndicators(scenario);
    return selectedIndicators;
  }, [viewMode, scenario, selectedIndicators]);

  const designHref = useMemo(() => {
    const params = scenarioToSearchParams(scenario ?? DEFAULT_SCENARIO);
    return `/design?${params.toString()}`;
  }, [scenario]);

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-8 py-16 text-center">
        <p className="text-ink-muted">{t("common.loading")}</p>
      </div>
    );
  }

  if (error || !result || !scenario || !executiveSummary) {
    return (
      <div className="max-w-7xl mx-auto px-8 py-16 text-center">
        <p className="text-negative mb-4">{error}</p>
        <Link href="/design" className="text-accent underline">
          {t("nav.design")}
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 md:px-8 py-8">
      <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-3 mb-2 flex-wrap">
            <span className="text-[11px] tracking-[0.2em] text-accent uppercase">{t("results.step")}</span>
            {sessionOnly && (
              <span className="text-[11px] px-2 py-0.5 bg-warning/15 text-warning rounded-md">
                {t("results.sessionSaved")}
              </span>
            )}
          </div>
          <h1 className="font-display text-3xl mb-1">{t("results.title")}</h1>
          <p className="text-ink-muted text-sm">
            {scenario.name} ·{" "}
            {t("common.yearsHorizon", {
              years: scenario.run.implementation_years + scenario.run.maintenance_years,
            })}{" "}
            · {t("common.vsBaseline")}
          </p>
        </div>
        <div className="flex gap-2 flex-wrap items-start justify-end">
          <BackToLastComparisonLink />
          <ResultsExportBar
            mode="scenario"
            scenario={scenario}
            result={result}
            runId={runId}
            scopeRef={exportScopeRef}
          />
          <Link
            href={designHref}
            className="px-4 py-2 border border-border rounded-md text-sm hover:bg-paper-deep"
          >
            {t("common.adjustScenario")}
          </Link>
        </div>
      </div>

      <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <ViewModeToggle mode={viewMode} onChange={setViewMode} />
        {viewMode === "analyst" && (
          <p className="text-xs text-ink-muted">{t("results.analystHint")}</p>
        )}
      </div>

      <CountyScopeBanner />

      {sessionOnly && (
        <div className="mb-6 text-sm bg-warning/10 border border-warning/30 rounded-lg px-4 py-3 text-ink-soft">
          {t("results.sessionSavedDetail")}
        </div>
      )}

      <ExecutiveSummary summary={executiveSummary} result={result} />

      <BudgetLens
        mode="single"
        label={scenario.name}
        costUsd={result.summary.cumulative_cost_usd}
      />

      <MethodsLimitations scenario={scenario} result={result} runId={runId} />

      {viewMode === "analyst" && (
        <IndicatorsDrawer
          scenario={scenario}
          selected={selectedIndicators}
          onChange={setSelectedIndicators}
        />
      )}

      <div ref={exportScopeRef}>
        <ResultsStories result={result} selectedIndicators={displayIndicators} />
      </div>

      <div className="mt-12 flex gap-4 flex-wrap items-center">
        <Link href={designHref} className="px-6 py-3 border border-border rounded-md hover:bg-paper-deep">
          ← {t("common.adjustScenario")}
        </Link>
        <BackToLastComparisonLink variant="text" />
        <Link href="/compare" className="px-6 py-3 bg-ink text-paper rounded-md">
          {t("common.newComparison")} →
        </Link>
      </div>
    </div>
  );
}

export default function ResultsPage() {
  return (
    <Suspense fallback={<div className="p-8">Loading…</div>}>
      <ResultsContent />
    </Suspense>
  );
}
