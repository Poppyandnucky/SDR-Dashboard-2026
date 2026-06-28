"use client";

import { useEffect, useRef, useState } from "react";
import { useCounty } from "@/components/county/CountyProvider";
import { useLocale } from "@/components/i18n/LocaleProvider";
import { COUNTIES } from "@/lib/counties";

export default function CountyDropdown() {
  const { t } = useLocale();
  const { countyId, county, setCountyId } = useCounty();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, []);

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 px-3 py-1.5 border border-border rounded-full hover:bg-paper-deep transition text-ink bg-card/80"
      >
        <svg viewBox="0 0 16 16" className="w-3.5 h-3.5 text-accent shrink-0" fill="none" aria-hidden>
          <path
            d="M8 1.5a4 4 0 0 0-4 4c0 3 4 8.5 4 8.5s4-5.5 4-8.5a4 4 0 0 0-4-4Z"
            stroke="currentColor"
            strokeWidth="1.3"
          />
          <circle cx="8" cy="5.5" r="1" fill="currentColor" stroke="none" />
        </svg>
        <span className="text-sm font-medium">{county.name}</span>
        <span className="text-ink-muted text-[10px]">▾</span>
      </button>
      {open && (
        <div className="absolute right-0 top-full mt-1 w-60 bg-card border border-border rounded-xl shadow-lg z-50 py-2 overflow-hidden">
          <div className="px-3 py-1.5 text-[10px] uppercase tracking-[0.15em] text-ink-muted">
            {t("start.selectCounty")}
          </div>
          {COUNTIES.map((c) => (
            <button
              key={c.id}
              type="button"
              className={`w-full text-left px-3 py-2.5 text-sm flex justify-between items-center gap-2 hover:bg-paper-deep/60 ${
                c.id === countyId ? "bg-paper-deep/40" : ""
              }`}
              onClick={() => {
                setCountyId(c.id);
                setOpen(false);
              }}
            >
              <span>{c.name}</span>
              {c.calibrated ? (
                <span className="text-[10px] text-intervention">{t("start.calibrated")}</span>
              ) : (
                <span className="text-[9px] text-ink-muted italic">
                  {t("start.comingSoonShort", { when: c.available ?? "soon" })}
                </span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
