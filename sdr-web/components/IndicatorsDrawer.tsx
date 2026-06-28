"use client";

import { useCallback, useMemo, useState } from "react";
import { useLocale } from "@/components/i18n/LocaleProvider";
import {
  getAllAvailableIndicators,
  getDefaultSelectedIndicators,
  getEssentialIndicators,
  indicatorsByStory,
  isIndicatorAvailable,
  INDICATOR_CATALOG,
  STORY_ORDER,
  StoryId,
} from "@/lib/indicators";
import { Scenario } from "@/lib/scenarios";

interface Props {
  scenario: Scenario;
  selected: Set<string>;
  onChange: (selected: Set<string>) => void;
}

const STORY_HINT_KEYS: Record<StoryId, string> = {
  kpi: "stories.kpiHint",
  story01: "stories.worthIt",
  story02: "stories.mothers",
  story03: "stories.delivery",
  story04: "stories.coping",
};

const STORY_LABEL_KEYS: Record<StoryId, string> = {
  kpi: "indicatorDrawer.storyKpi",
  story01: "stories.story01",
  story02: "stories.story02",
  story03: "stories.story03",
  story04: "stories.story04",
};

export default function IndicatorsDrawer({ scenario, selected, onChange }: Props) {
  const { t } = useLocale();
  const [open, setOpen] = useState(false);

  const available = useMemo(() => getAllAvailableIndicators(scenario), [scenario]);
  const selectedCount = useMemo(
    () => INDICATOR_CATALOG.filter((i) => selected.has(i.id)).length,
    [selected]
  );

  const toggle = useCallback(
    (id: string) => {
      const next = new Set(selected);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      onChange(next);
    },
    [selected, onChange]
  );

  const applyDefaults = () => onChange(getDefaultSelectedIndicators(scenario));
  const applyEssentials = () => onChange(getEssentialIndicators(scenario));
  const applyAll = () => onChange(new Set(available));

  return (
    <details
      className="mb-10 bg-card border border-border rounded-xl overflow-hidden group"
      open={open}
      onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}
    >
      <summary className="px-7 py-4 cursor-pointer hover:bg-paper-deep/40 flex items-center justify-between transition list-none">
        <div className="flex items-center gap-4">
          <span
            className={`text-ink-muted text-sm transition ${open ? "rotate-180" : ""}`}
          >
            ▾
          </span>
          <div>
            <div className="flex items-center gap-3 flex-wrap">
              <span className="text-[10px] tracking-[0.18em] text-ink-muted uppercase font-medium">
                {t("indicatorDrawer.customize")}
              </span>
              <span className="font-display text-lg">{t("indicatorDrawer.title")}</span>
              <span className="num text-[11px] px-2 py-0.5 bg-ink text-paper rounded-md">
                {t("indicatorDrawer.selectedCount", {
                  count: selectedCount,
                  total: INDICATOR_CATALOG.length,
                })}
              </span>
            </div>
            <div className="text-[11px] text-ink-muted mt-0.5">
              {t("indicatorDrawer.subtitle")}
            </div>
          </div>
        </div>
        <span className="text-[11px] text-ink-muted italic hidden sm:inline">
          {open ? t("indicatorDrawer.collapse") : t("indicatorDrawer.expand")}
        </span>
      </summary>

      <div className="border-t border-border-soft">
        <div className="px-7 py-3 bg-warning/5 border-b border-border-soft text-[11px] text-ink-soft">
          {t("indicatorDrawer.disclaimer")}
        </div>

        <div className="divide-y divide-border-soft">
          {STORY_ORDER.map((story) => {
            const items = indicatorsByStory(story);
            if (items.length === 0) return null;

            return (
              <div key={story} className="p-5 md:px-7">
                <div className="mb-3">
                  <h4 className="font-display text-base text-ink">{t(STORY_LABEL_KEYS[story])}</h4>
                  <p className="text-[11px] text-ink-muted mt-0.5">{t(STORY_HINT_KEYS[story])}</p>
                </div>
                <ul className="space-y-3">
                  {items.map((ind) => {
                    const avail = isIndicatorAvailable(ind, scenario);
                    const checked = selected.has(ind.id);
                    return (
                      <li key={ind.id}>
                        <label
                          className={`flex items-start gap-2.5 ${avail ? "cursor-pointer group" : "opacity-40 cursor-not-allowed"}`}
                        >
                          <input
                            type="checkbox"
                            checked={checked}
                            disabled={!avail}
                            onChange={() => avail && toggle(ind.id)}
                            className="w-3.5 h-3.5 accent-intervention mt-0.5 shrink-0"
                          />
                          <span className="min-w-0">
                            <span
                              className={`block text-sm ${checked ? "text-ink" : "text-ink-soft"} group-hover:text-ink`}
                            >
                              {ind.name}
                            </span>
                            <span className="block text-[11px] text-ink-muted mt-0.5 leading-snug">
                              {t("indicatorDrawer.addsPrefix")}{" "}
                              {t(`indicatorAdds.${ind.addsKey}`)}
                              {!ind.wired && (
                                <span className="ml-1.5 text-[10px] uppercase tracking-wide text-ink-muted/80">
                                  · {t("indicatorDrawer.catalogOnly")}
                                </span>
                              )}
                            </span>
                          </span>
                        </label>
                      </li>
                    );
                  })}
                </ul>
              </div>
            );
          })}
        </div>

        <div className="px-7 py-3 bg-paper/40 border-t border-border-soft flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-2 flex-wrap">
            <button
              type="button"
              onClick={applyEssentials}
              className="text-[12px] text-ink-soft hover:text-ink px-2 py-1"
            >
              {t("indicatorDrawer.selectEssentials")}
            </button>
            <span className="text-ink-muted">·</span>
            <button
              type="button"
              onClick={applyAll}
              className="text-[12px] text-ink-soft hover:text-ink px-2 py-1"
            >
              {t("indicatorDrawer.selectAll")}
            </button>
            <span className="text-ink-muted">·</span>
            <button
              type="button"
              onClick={applyDefaults}
              className="text-[12px] text-accent hover:underline px-2 py-1 font-medium"
            >
              {t("indicatorDrawer.defaults")}
            </button>
          </div>
          <div className="text-[11px] text-ink-muted flex items-center gap-4">
            <span className="flex items-center gap-1.5">
              <input type="checkbox" checked readOnly className="w-3 h-3" disabled />{" "}
              {t("indicatorDrawer.legendSelected")}
            </span>
            <span className="flex items-center gap-1.5">
              <input type="checkbox" readOnly className="w-3 h-3" disabled />{" "}
              {t("indicatorDrawer.legendNotShown")}
            </span>
          </div>
        </div>
      </div>
    </details>
  );
}
