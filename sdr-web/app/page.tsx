"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { useCounty } from "@/components/county/CountyProvider";
import { useLocale } from "@/components/i18n/LocaleProvider";
import KenyaCountyMap from "@/components/landing/KenyaCountyMap";
import { fetchPresets } from "@/lib/api";
import { getPresetDisplay } from "@/lib/preset-labels";
import { Preset } from "@/lib/scenarios";
import { scenarioToSearchParams } from "@/lib/url-state";

export default function StartPage() {
  const { t } = useLocale();
  const { county } = useCounty();
  const [presets, setPresets] = useState<Preset[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPresets()
      .then(setPresets)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const featuredPreset = useMemo(
    () => presets.find((p) => p.id === "combined") ?? presets[0],
    [presets]
  );

  const countyLabel = county.population
    ? t("start.countyActive", {
        name: county.name,
        pop: (county.population / 1_000_000).toFixed(2),
      })
    : t("start.county");

  return (
    <div className="landing-page mx-auto max-w-7xl px-4 md:px-8">
      {/* Hero + map */}
      <section className="grid lg:grid-cols-[1fr_0.85fr] lg:gap-6 lg:flex-1 lg:min-h-0 lg:items-stretch gap-8 mb-8 lg:mb-3">
        <div className="flex flex-col justify-center min-h-0 lg:py-1">
          <p className="text-[10px] uppercase tracking-[0.2em] text-accent mb-2">{countyLabel}</p>
          <h1 className="font-display text-3xl xl:text-[2.35rem] font-medium leading-[1.1] mb-3 max-w-xl">
            {t("start.hero")}
          </h1>
          <p className="text-ink-soft text-sm xl:text-base leading-relaxed max-w-lg mb-4 line-clamp-3 lg:line-clamp-2">
            {t("start.lead")}
          </p>

          <div className="inline-flex items-center gap-2.5 rounded-lg border border-border bg-card px-3 py-2 mb-4 w-fit">
            <span className="font-display text-2xl num text-accent">{t("start.mmrToday")}</span>
            <p className="text-xs text-ink-muted max-w-[11rem] leading-snug">{t("start.mmrLabel")}</p>
          </div>

          {featuredPreset && (
            <div className="flex flex-wrap items-center gap-2.5">
              <Link
                href={`/design?${scenarioToSearchParams(featuredPreset.scenario).toString()}`}
                className="inline-flex items-center gap-2 px-5 py-2.5 bg-ink text-paper rounded-lg text-sm font-medium hover:opacity-90 transition shadow-sm"
              >
                {t("start.featuredCta")}
                <span aria-hidden>→</span>
              </Link>
              <Link
                href="/design"
                className="text-xs text-ink-muted hover:text-accent underline underline-offset-4 transition"
              >
                {t("start.custom")}
              </Link>
            </div>
          )}
        </div>

        <div className="min-h-[220px] sm:min-h-[260px] lg:min-h-0 lg:h-full">
          <KenyaCountyMap compact />
        </div>
      </section>

      {/* Presets */}
      <section className="lg:shrink-0 lg:pb-1">
        <div className="mb-3 flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <h2 className="font-display text-lg">{t("start.presets")}</h2>
          <p className="text-xs text-ink-muted">{t("start.presetsHint")}</p>
        </div>

        {loading ? (
          <p className="text-sm text-ink-muted">{t("start.loadingPresets")}</p>
        ) : (
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {presets.map((preset) => {
              const display = getPresetDisplay(preset.id, t);
              return (
                <Link
                  key={preset.id}
                  href={`/design?${scenarioToSearchParams(preset.scenario).toString()}`}
                  className="preset-card group block bg-card border border-border rounded-xl p-4 lg:p-3.5 xl:p-4"
                >
                  <div className="flex items-start justify-between gap-2 mb-0.5">
                    <h3 className="font-display text-sm xl:text-base leading-snug line-clamp-2">
                      {display?.name ?? preset.name}
                    </h3>
                    <span className="preset-card-arrow text-accent shrink-0 text-sm" aria-hidden>
                      →
                    </span>
                  </div>
                  <p className="text-[11px] text-ink-muted mb-1 line-clamp-1">
                    {display?.subtitle ?? preset.subtitle}
                  </p>
                  <p className="text-xs text-ink-soft leading-snug line-clamp-2">
                    {display?.description ?? preset.description}
                  </p>
                </Link>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
