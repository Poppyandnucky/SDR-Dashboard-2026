"use client";

import { HSSIntensity } from "@/lib/scenarios";
import {
  hasIntervention,
  INTERVENTION_LIBRARY,
  InterventionId,
  LibraryItem,
  removeIntervention,
  setHssIntensity,
} from "@/lib/interventions";
import PillSelector from "@/components/PillSelector";

const HSS_OPTIONS = [
  { value: "light", label: "Light" },
  { value: "moderate", label: "Moderate" },
  { value: "intensive", label: "Intensive" },
];

interface Props {
  id: InterventionId;
  hssIntensity?: HSSIntensity;
  onIntensityChange?: (intensity: HSSIntensity) => void;
  onRemove: () => void;
}

function getItem(id: InterventionId): LibraryItem {
  return INTERVENTION_LIBRARY.find((i) => i.id === id)!;
}

export default function InterventionCard({
  id,
  hssIntensity,
  onIntensityChange,
  onRemove,
}: Props) {
  const item = getItem(id);

  return (
    <div className="bg-card border border-border rounded-lg p-4">
      <div className="flex items-start justify-between mb-1">
        <div>
          <h4 className="text-sm font-medium flex items-center gap-2">
            {item.name}
            {item.wired === "ui-only" && (
              <span className="text-[9px] text-warning">● UI only</span>
            )}
            {item.wired === "partial" && (
              <span className="text-[9px] text-warning">● partial</span>
            )}
          </h4>
          <p className="text-[11px] text-ink-muted mt-0.5">{item.description}</p>
        </div>
        <button
          type="button"
          onClick={onRemove}
          className="text-ink-muted hover:text-ink text-lg leading-none px-1"
          aria-label={`Remove ${item.name}`}
        >
          ×
        </button>
      </div>

      {id === "hss" && onIntensityChange && (
        <div className="mt-3">
          <PillSelector
            options={HSS_OPTIONS}
            value={hssIntensity ?? "moderate"}
            onChange={(v) => onIntensityChange(v as HSSIntensity)}
          />
        </div>
      )}

      {id === "fqa" && (
        <p className="text-[10px] text-ink-muted mt-2 italic">
          Stored on scenario; not yet wired into simulation.
        </p>
      )}
      {id === "pulse" && (
        <p className="text-[10px] text-ink-muted mt-2 italic">
          Stored on scenario; not yet wired into simulation.
        </p>
      )}
    </div>
  );
}

export { getItem, hasIntervention, removeIntervention, setHssIntensity };
