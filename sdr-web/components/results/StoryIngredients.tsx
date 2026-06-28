"use client";

import { useLocale } from "@/components/i18n/LocaleProvider";
import { getStoryIngredients, StoryId } from "@/lib/indicators";

interface Props {
  story: StoryId;
  selected: Set<string>;
}

export default function StoryIngredients({ story, selected }: Props) {
  const { t } = useLocale();
  const modules = getStoryIngredients(story, selected);

  if (modules.length === 0) return null;

  return (
    <div className="flex flex-wrap items-center gap-2 mb-4">
      <span className="text-[10px] uppercase tracking-widest text-ink-muted shrink-0">
        {t("indicatorDrawer.includes")}
      </span>
      {modules.map((mod) => (
        <span
          key={mod.id}
          className={`text-[11px] px-2 py-0.5 rounded-md border ${
            mod.wired
              ? "bg-paper-deep/60 border-border-soft text-ink-soft"
              : "border-dashed border-ink-muted/40 text-ink-muted"
          }`}
          title={mod.wired ? undefined : t("indicatorDrawer.catalogOnly")}
        >
          {t(`indicatorModules.${mod.moduleKey}`)}
          {!mod.wired && (
            <span className="ml-1 opacity-70" aria-hidden>
              ·
            </span>
          )}
        </span>
      ))}
    </div>
  );
}
