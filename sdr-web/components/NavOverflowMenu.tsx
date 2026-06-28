"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { useCounty } from "@/components/county/CountyProvider";
import { useLocale } from "@/components/i18n/LocaleProvider";
import { COUNTIES } from "@/lib/counties";

interface Props {
  compareHref?: string;
  showCompare?: boolean;
  onShare?: () => void;
}

export default function NavOverflowMenu({ compareHref, showCompare, onShare }: Props) {
  const { t } = useLocale();
  const { countyId, setCountyId } = useCounty();
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
        onClick={() => setOpen((v) => !v)}
        className="p-2 rounded-md border border-border text-ink-muted hover:text-ink hover:bg-paper-deep transition"
        aria-expanded={open}
        aria-haspopup="menu"
        aria-label={t("nav.menu")}
      >
        <svg viewBox="0 0 20 20" className="w-4 h-4" fill="currentColor" aria-hidden>
          <circle cx="4" cy="10" r="1.5" />
          <circle cx="10" cy="10" r="1.5" />
          <circle cx="16" cy="10" r="1.5" />
        </svg>
      </button>

      {open && (
        <div
          role="menu"
          className="absolute right-0 top-full mt-1 w-56 bg-card border border-border rounded-xl shadow-lg z-50 py-1 overflow-hidden"
        >
          <div className="px-3 py-2 text-[10px] uppercase tracking-[0.15em] text-ink-muted border-b border-border-soft">
            {t("nav.countyScope")}
          </div>
          {COUNTIES.map((c) => (
            <button
              key={c.id}
              type="button"
              role="menuitem"
              className={`w-full text-left px-3 py-2 text-sm flex justify-between items-center gap-2 hover:bg-paper-deep/60 ${
                c.id === countyId ? "bg-paper-deep/40 font-medium" : ""
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

          <div className="border-t border-border-soft my-1" />

          {showCompare && compareHref && (
            <Link
              href={compareHref}
              role="menuitem"
              className="block px-3 py-2 text-sm hover:bg-paper-deep/60"
              onClick={() => setOpen(false)}
            >
              {t("nav.compare")}
            </Link>
          )}
          <Link
            href="/about"
            role="menuitem"
            className="block px-3 py-2 text-sm hover:bg-paper-deep/60"
            onClick={() => setOpen(false)}
          >
            {t("nav.about")}
          </Link>
          {onShare && (
            <button
              type="button"
              role="menuitem"
              className="w-full text-left px-3 py-2 text-sm hover:bg-paper-deep/60"
              onClick={() => {
                setOpen(false);
                onShare();
              }}
            >
              {t("nav.share")}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
