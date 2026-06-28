"use client";

import InterventionCard from "@/components/compare/InterventionCard";
import {
  applyIntervention,
  hasIntervention,
  listActiveInterventions,
  removeIntervention,
  setHssIntensity,
} from "@/lib/interventions";
import { HSSIntensity, Scenario } from "@/lib/scenarios";

type ColumnAccent = "a" | "b";

const ACCENT: Record<
  ColumnAccent,
  { border: string; headerBg: string; headerText: string }
> = {
  a: {
    border: "#C0D5E8",
    headerBg: "#E0EBF5",
    headerText: "#2563A8",
  },
  b: {
    border: "#BFDEC4",
    headerBg: "#DCEEE0",
    headerText: "#2B7A3E",
  },
};

interface Props {
  accent: ColumnAccent;
  scenario: Scenario;
  onChange: (scenario: Scenario) => void;
}

export default function ScenarioColumn({ accent, scenario, onChange }: Props) {
  const colors = ACCENT[accent];
  const activeIds = listActiveInterventions(scenario);

  return (
    <div
      className="rounded-xl border-2"
      style={{ borderColor: colors.border, background: accent === "a" ? "#F4F8FC" : "#F2FAF4" }}
    >
      <div
        className="px-5 py-4 border-b"
        style={{ borderColor: colors.border, background: colors.headerBg }}
      >
        <input
          type="text"
          value={scenario.name}
          onChange={(e) => onChange({ ...scenario, name: e.target.value })}
          className="font-display text-lg bg-transparent border-none outline-none w-full"
          style={{ color: colors.headerText }}
        />
        <span className="text-[10px] text-ink-muted italic">Click name to edit</span>
      </div>

      <div className="p-4 space-y-3 min-h-[200px]">
        {activeIds.length === 0 ? (
          <p className="text-sm text-ink-muted text-center py-8">
            No interventions yet — add from the library using + {accent === "a" ? "A" : "B"}
          </p>
        ) : (
          activeIds.map((id) => (
            <InterventionCard
              key={id}
              id={id}
              hssIntensity={scenario.hss.intensity}
              onIntensityChange={
                id === "hss"
                  ? (intensity) => onChange(setHssIntensity(scenario, intensity))
                  : undefined
              }
              onRemove={() => onChange(removeIntervention(scenario, id))}
            />
          ))
        )}
      </div>
    </div>
  );
}

export { applyIntervention, hasIntervention };
