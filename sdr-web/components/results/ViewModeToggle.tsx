"use client";

import { ResultsViewMode } from "@/lib/results-summary";
import { useLocale } from "@/components/i18n/LocaleProvider";

interface Props {
  mode: ResultsViewMode;
  onChange: (mode: ResultsViewMode) => void;
}

export default function ViewModeToggle({ mode, onChange }: Props) {
  const { t } = useLocale();

  return (
    <div
      className="inline-flex rounded-lg border border-border bg-paper-deep p-0.5"
      role="radiogroup"
      aria-label={t("results.policyBrief")}
    >
      <ViewOption
        active={mode === "policy"}
        label={t("viewMode.policy")}
        hint={t("viewMode.policyHint")}
        onClick={() => onChange("policy")}
      />
      <ViewOption
        active={mode === "analyst"}
        label={t("viewMode.analyst")}
        hint={t("viewMode.analystHint")}
        onClick={() => onChange("analyst")}
      />
    </div>
  );
}

function ViewOption({
  active,
  label,
  hint,
  onClick,
}: {
  active: boolean;
  label: string;
  hint: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={active}
      onClick={onClick}
      className={`px-3 py-1.5 rounded-md text-left transition ${
        active ? "bg-card shadow-sm text-ink" : "text-ink-muted hover:text-ink-soft"
      }`}
    >
      <span className="block text-xs font-medium">{label}</span>
      <span className="block text-[10px] opacity-70">{hint}</span>
    </button>
  );
}
