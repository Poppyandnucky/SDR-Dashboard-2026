"use client";

import {
  applyIntervention,
  GROUP_LABELS,
  hasIntervention,
  INTERVENTION_LIBRARY,
  InterventionGroup,
  InterventionId,
  LibraryItem,
} from "@/lib/interventions";
import { Scenario } from "@/lib/scenarios";

type ColumnTarget = "a" | "b";

interface Props {
  scenarioA: Scenario;
  scenarioB: Scenario;
  onAdd: (target: ColumnTarget, id: InterventionId) => void;
}

const GROUPS: InterventionGroup[] = ["supply", "treatments", "community"];

function LibButton({
  active,
  label,
  onClick,
  color,
}: {
  active: boolean;
  label: string;
  onClick: () => void;
  color: "blue" | "green" | "neutral";
}) {
  const styles =
    color === "blue"
      ? { background: "#E0EBF5", color: "#2563A8", border: "1px solid #C0D5E8" }
      : color === "green"
        ? { background: "#DCEEE0", color: "#2B7A3E", border: "1px solid #BFDEC4" }
        : {};

  return (
    <button
      type="button"
      onClick={onClick}
      className={`px-3 py-1 rounded text-[11px] flex items-center justify-center gap-1 border border-border text-ink-soft hover:bg-paper-deep ${
        active ? "" : ""
      }`}
      style={active ? styles : undefined}
    >
      {active ? "✓" : "+"} {label}
    </button>
  );
}

function wiredBadge(item: LibraryItem) {
  if (item.wired === "wired") return <span className="text-[9px] text-positive ml-1">● wired</span>;
  if (item.wired === "ui-only")
    return <span className="text-[9px] text-warning ml-1">● UI only</span>;
  return <span className="text-[9px] text-warning ml-1">● partial</span>;
}

export default function InterventionLibrary({ scenarioA, scenarioB, onAdd }: Props) {
  return (
    <aside className="sticky top-28">
      <h2 className="font-display text-2xl leading-tight mb-1">Intervention Library</h2>
      <p className="text-xs text-ink-muted mb-5">
        Click <span className="num">+ A</span> or <span className="num">+ B</span> to add to a
        scenario.
      </p>

      {GROUPS.map((group) => {
        const meta = GROUP_LABELS[group];
        const items = INTERVENTION_LIBRARY.filter((i) => i.group === group);
        return (
          <div key={group} className="mb-5">
            <div className="flex items-center gap-2 mb-2.5">
              <span className={`w-2 h-2 rounded-full ${meta.dot}`} />
              <span className="text-sm font-medium">{meta.label}</span>
            </div>
            <div className="space-y-3 pl-4">
              {items.map((item) => {
                const inA = hasIntervention(scenarioA, item.id);
                const inB = hasIntervention(scenarioB, item.id);
                return (
                  <div key={item.id}>
                    <div className="text-[12px] mb-1.5">
                      {item.name}
                      {item.group === "community" && wiredBadge(item)}
                    </div>
                    <div className="grid grid-cols-2 gap-1.5">
                      <LibButton
                        active={inA}
                        label="A"
                        color="blue"
                        onClick={() => onAdd("a", item.id)}
                      />
                      <LibButton
                        active={inB}
                        label="B"
                        color="green"
                        onClick={() => onAdd("b", item.id)}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}

      <div className="mt-3 pl-4 pt-2 border-t border-border-soft text-[9px] text-ink-muted leading-relaxed">
        <span className="text-positive">●</span> drives simulation ·{" "}
        <span className="text-warning">●</span> UI controls only (model wiring pending)
      </div>
    </aside>
  );
}

export { applyIntervention };
