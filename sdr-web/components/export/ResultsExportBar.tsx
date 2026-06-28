"use client";

import { RefObject, useState } from "react";
import { useLocale } from "@/components/i18n/LocaleProvider";
import { exportComparisonResultsPdf, exportScenarioResultsPdf } from "@/lib/results-pdf-export";
import { CompareResponse, Scenario, ScenarioResult } from "@/lib/scenarios";

interface ScenarioProps {
  mode: "scenario";
  scenario: Scenario;
  result: ScenarioResult;
  runId?: string | null;
  scopeRef: RefObject<HTMLElement | null>;
}

interface CompareProps {
  mode: "compare";
  data: CompareResponse;
  scopeRef: RefObject<HTMLElement | null>;
}

type Props = ScenarioProps | CompareProps;

export default function ResultsExportBar(props: Props) {
  const { locale, t } = useLocale();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePdf = async () => {
    const scope = props.scopeRef.current;
    if (!scope) {
      setError("Nothing to export yet.");
      return;
    }

    setBusy(true);
    setError(null);
    try {
      if (props.mode === "scenario") {
        await exportScenarioResultsPdf(props.scenario, props.result, scope, props.runId, locale);
      } else {
        await exportComparisonResultsPdf(props.data, scope, locale);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "PDF export failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col items-end gap-1">
      <button
        type="button"
        onClick={handlePdf}
        disabled={busy}
        className="px-4 py-2 border border-border rounded-md text-sm hover:bg-paper-deep disabled:opacity-50"
      >
        {busy ? t("results.buildingPdf") : t("results.downloadPdf")}
      </button>
      {error && <p className="text-xs text-negative max-w-xs text-right">{error}</p>}
    </div>
  );
}
