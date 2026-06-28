"use client";

import { useEffect, useMemo, useState } from "react";
import { useLocale } from "@/components/i18n/LocaleProvider";
import {
  BUDGET_PRESETS,
  clearStoredBudgetCap,
  evaluateBudget,
  formatUsd,
  getStoredBudgetCap,
  storeBudgetCap,
} from "@/lib/budget-lens";

interface SingleCostProps {
  mode: "single";
  label: string;
  costUsd: number;
}

interface CompareCostProps {
  mode: "compare";
  labelA: string;
  labelB: string;
  costAUsd: number;
  costBUsd: number;
}

type Props = SingleCostProps | CompareCostProps;

export default function BudgetLens(props: Props) {
  const { t } = useLocale();
  const [capUsd, setCapUsd] = useState<number | null>(null);
  const [customUsd, setCustomUsd] = useState("");
  const [enabled, setEnabled] = useState(false);

  useEffect(() => {
    const stored = getStoredBudgetCap();
    if (stored) {
      setCapUsd(stored);
      setEnabled(true);
      setCustomUsd(String(stored));
    }
  }, []);

  const persistCap = (value: number) => {
    setCapUsd(value);
    setEnabled(true);
    storeBudgetCap(value);
  };

  const evaluation = useMemo(() => {
    if (!enabled || !capUsd) return null;
    if (props.mode === "single") {
      return { single: evaluateBudget(props.costUsd, capUsd) };
    }
    return {
      a: evaluateBudget(props.costAUsd, capUsd),
      b: evaluateBudget(props.costBUsd, capUsd),
    };
  }, [enabled, capUsd, props]);

  const applyCustom = () => {
    const n = Number(customUsd.replace(/,/g, ""));
    if (Number.isFinite(n) && n > 0) persistCap(n);
  };

  return (
    <section className="mb-8 bg-card border border-border rounded-xl p-6">
      <div className="flex flex-wrap items-start justify-between gap-4 mb-4">
        <div>
          <div className="text-[11px] uppercase tracking-[0.2em] text-accent mb-1">
            {t("budget.title")}
          </div>
          <h2 className="font-display text-lg">{t("budget.heading")}</h2>
          <p className="text-xs text-ink-muted mt-1 max-w-xl">{t("budget.hint")}</p>
        </div>
        {enabled && (
          <button
            type="button"
            onClick={() => {
              setEnabled(false);
              setCapUsd(null);
              clearStoredBudgetCap();
            }}
            className="text-xs text-ink-muted hover:text-ink underline"
          >
            {t("budget.clear")}
          </button>
        )}
      </div>

      <div className="flex flex-wrap gap-2 mb-4">
        {BUDGET_PRESETS.map((preset) => (
          <button
            key={preset.value}
            type="button"
            onClick={() => {
              persistCap(preset.value);
              setCustomUsd(String(preset.value));
            }}
            className={`px-3 py-1.5 text-xs rounded-md border transition ${
              capUsd === preset.value
                ? "bg-ink text-paper border-ink"
                : "border-border hover:bg-paper-deep"
            }`}
          >
            {preset.label}
          </button>
        ))}
      </div>

      <div className="flex flex-wrap items-end gap-2 mb-4">
        <label className="text-xs text-ink-muted">
          {t("budget.customCap")}
          <input
            type="text"
            inputMode="numeric"
            value={customUsd}
            onChange={(e) => setCustomUsd(e.target.value)}
            placeholder="e.g. 8000000"
            className="block mt-1 w-40 border border-border rounded-md px-3 py-1.5 text-sm bg-paper-deep"
          />
        </label>
        <button
          type="button"
          onClick={applyCustom}
          className="px-3 py-1.5 text-xs border border-border rounded-md hover:bg-paper-deep"
        >
          {t("budget.apply")}
        </button>
      </div>

      {!evaluation && <p className="text-sm text-ink-muted">{t("budget.selectCap")}</p>}

      {evaluation?.single && (
        <BudgetStatus
          label={props.mode === "single" ? props.label : ""}
          eval={evaluation.single}
          t={t}
        />
      )}

      {evaluation?.a && evaluation?.b && props.mode === "compare" && (
        <div className="grid md:grid-cols-2 gap-4">
          <BudgetStatus label={props.labelA} eval={evaluation.a} accent="#2563A8" t={t} />
          <BudgetStatus label={props.labelB} eval={evaluation.b} accent="#2B7A3E" t={t} />
        </div>
      )}
    </section>
  );
}

function BudgetStatus({
  label,
  eval: ev,
  accent,
  t,
}: {
  label: string;
  eval: ReturnType<typeof evaluateBudget>;
  accent?: string;
  t: (key: string, params?: Record<string, string | number>) => string;
}) {
  const detail = ev.withinBudget
    ? t("budget.headroom", {
        amount: formatUsd(ev.headroomUsd),
        pct: ev.pctOfBudget.toFixed(0),
      })
    : t("budget.overrun", {
        amount: formatUsd(ev.overByUsd),
        pct: ev.pctOfBudget.toFixed(0),
      });

  return (
    <div
      className={`rounded-lg border px-4 py-3 ${
        ev.withinBudget ? "border-intervention/30 bg-intervention-soft/30" : "border-warning/40 bg-warning/10"
      }`}
    >
      {label && (
        <div className="text-xs font-medium mb-2" style={accent ? { color: accent } : undefined}>
          {label}
        </div>
      )}
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="text-sm">
          {t("budget.projected")}{" "}
          <strong className="num">{formatUsd(ev.costUsd)}</strong>
        </span>
        <span
          className={`text-xs font-medium px-2 py-0.5 rounded ${
            ev.withinBudget ? "bg-intervention-soft text-intervention" : "bg-warning/20 text-warning"
          }`}
        >
          {ev.withinBudget ? t("budget.within") : t("budget.over")}
        </span>
      </div>
      <p className="text-xs text-ink-muted mt-2 num">
        {t("budget.capLine", { cap: formatUsd(ev.capUsd), detail })}
      </p>
    </div>
  );
}
